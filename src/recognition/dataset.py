"""PyTorch Dataset adapter and collate function for IAM-line handwriting dataset."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from src.recognition.preprocessing import LineImagePreprocessor
from src.recognition.vocabulary import Vocabulary


class IAMLineDataset(Dataset):
    """Adapter wrapping Hugging Face Teklia/IAM-line dataset for PyTorch training and evaluation."""

    def __init__(
        self,
        hf_split_dataset,
        vocabulary: Vocabulary,
        preprocessor: Optional[LineImagePreprocessor] = None,
        image_col: str = "image",
        text_col: str = "text",
        is_training: bool = False,
    ):
        """Initialize IAMLineDataset.

        Args:
            hf_split_dataset: A split from load_dataset('Teklia/IAM-line').
            vocabulary: Character vocabulary mapping characters to indices.
            preprocessor: LineImagePreprocessor instance.
            image_col: Name of image column in HF dataset (default: 'image').
            text_col: Name of transcription column in HF dataset (default: 'text').
            is_training: Whether training augmentations should be enabled.
        """
        self.dataset = hf_split_dataset
        self.vocabulary = vocabulary
        self.preprocessor = preprocessor or LineImagePreprocessor()
        self.image_col = image_col
        self.text_col = text_col
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load and preprocess a single sample.

        Returns:
            Dict containing:
                - 'image': torch.Tensor of shape (1, 32, W)
                - 'target': torch.Tensor of integer character indices
                - 'raw_text': str ground-truth transcription
                - 'input_length': int estimated time-steps after CNN (W // 4)
                - 'target_length': int number of characters in transcription
        """
        item = self.dataset[idx]
        raw_img = item[self.image_col]
        raw_text = str(item[self.text_col] or "").strip()

        # Preprocess image into normalized float32 tensor
        tensor_img = self.preprocessor.preprocess(raw_img, augment=self.is_training)
        img_width = tensor_img.shape[2]

        # In our CNN architecture, horizontal width is downsampled by factor of 4:
        # Time steps T = img_width // 4
        input_len = img_width // 4

        # Encode target transcription
        encoded_target = self.vocabulary.encode(raw_text)
        target_tensor = torch.tensor(encoded_target, dtype=torch.long)
        target_len = len(encoded_target)

        return {
            "image": tensor_img,
            "target": target_tensor,
            "raw_text": raw_text,
            "input_length": input_len,
            "target_length": target_len,
        }


def iam_collate_fn(
    batch: List[Dict[str, Any]],
    cnn_downsample_factor: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Custom batch collate function for variable-width line images with CTC targets.

    Filters samples with invalid CTC alignments (target_length > input_length),
    pads images along width to the maximum width in the batch, and concatenates 1D targets.

    Args:
        batch: List of sample dicts from IAMLineDataset.
        cnn_downsample_factor: Ratio by which CNN pooling reduces horizontal width.

    Returns:
        Tuple of:
            - padded_images: (Batch, 1, 32, Max_Width)
            - concatenated_targets: 1D Tensor of all targets in batch
            - input_lengths: 1D Tensor (Batch,) of time steps per sample
            - target_lengths: 1D Tensor (Batch,) of target lengths per sample
            - raw_texts: List of ground-truth strings
    """
    # Filter any degenerate sample where target_length > input_length
    valid_samples: List[Dict[str, Any]] = []
    for s in batch:
        if s["target_length"] > 0 and s["input_length"] >= s["target_length"]:
            valid_samples.append(s)

    # Fallback to keep at least 1 sample if filtering emptied the batch
    if not valid_samples:
        valid_samples = [batch[0]]

    batch_size = len(valid_samples)
    max_w = max(s["image"].shape[2] for s in valid_samples)

    # Ensure max_w is divisible by cnn_downsample_factor for clean CNN feature maps
    if max_w % cnn_downsample_factor != 0:
        max_w += (cnn_downsample_factor - (max_w % cnn_downsample_factor))

    # Initialize padded image tensor with zeros (background value after inversion)
    target_height = valid_samples[0]["image"].shape[1]
    padded_images = torch.zeros((batch_size, 1, target_height, max_w), dtype=torch.float32)

    input_lengths_list: List[int] = []
    target_lengths_list: List[int] = []
    targets_list: List[torch.Tensor] = []
    raw_texts: List[str] = []

    for i, s in enumerate(valid_samples):
        w = s["image"].shape[2]
        padded_images[i, :, :, :w] = s["image"]
        # CNN has stage 7 unpadded Conv2d(512, 512, 2), making output time length (w // 4) - 1
        input_lengths_list.append(max(1, (w // cnn_downsample_factor) - 1))
        target_lengths_list.append(s["target_length"])
        targets_list.append(s["target"])
        raw_texts.append(s["raw_text"])

    concatenated_targets = torch.cat(targets_list, dim=0) if targets_list else torch.tensor([], dtype=torch.long)
    input_lengths = torch.tensor(input_lengths_list, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths_list, dtype=torch.long)

    return padded_images, concatenated_targets, input_lengths, target_lengths, raw_texts
