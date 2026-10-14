"""PyTorch Dataset and collator for handwritten line crops."""

from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.ocr.decoder import CTCDecoder
from src.preprocessing.image_processor import ImageProcessor


class LineCropDataset(Dataset):
    """Dataset for training and validating CRNN on line image crops and transcriptions."""

    def __init__(
        self,
        samples: List[Tuple[np.ndarray, str]],
        decoder: CTCDecoder,
        processor: ImageProcessor,
    ):
        """Initialize Dataset.

        Args:
            samples (List[Tuple[np.ndarray, str]]): Tuples of (image_array, transcript).
            decoder (CTCDecoder): CTC decoder for encoding transcript to integers.
            processor (ImageProcessor): Preprocessor for normalizing line crops.
        """
        self.samples = samples
        self.decoder = decoder
        self.processor = processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        image, label_text = self.samples[idx]
        norm_img = self.processor.normalize_line_image(image)

        # Shape (1, 32, W)
        tensor_img = torch.tensor(norm_img, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(self.decoder.encode(label_text), dtype=torch.long)

        input_len = tensor_img.shape[2]  # Width
        target_len = len(target)

        return tensor_img, target, input_len, target_len


def pad_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int, int]]):
    """Collate function that dynamically pads variable-width line images in a batch.

    Args:
        batch: List of (tensor_img, target, input_len, target_len).

    Returns:
        Padded batch ready for CTCLoss.
    """
    imgs, targets, input_lens, target_lens = zip(*batch)

    # Determine maximum width in this batch
    max_w = max(img.shape[2] for img in imgs)
    # Ensure minimum width multiple of 4 for CNN pooling layers
    max_w = max(max_w, 32)
    if max_w % 4 != 0:
        max_w += (4 - (max_w % 4))

    batch_size = len(imgs)
    padded_imgs = torch.ones((batch_size, 1, 32, max_w), dtype=torch.float32)

    for i, img in enumerate(imgs):
        w = img.shape[2]
        padded_imgs[i, :, :, :w] = img

    concatenated_targets = torch.cat(targets, dim=0)
    target_lengths = torch.tensor(target_lens, dtype=torch.long)
    # The CNN architecture downsamples width by a factor of 4: W -> W / 4
    feature_lengths = torch.tensor([img.shape[2] // 4 for img in imgs], dtype=torch.long)

    return padded_imgs, concatenated_targets, feature_lengths, target_lengths
