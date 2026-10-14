"""Inference service for handwritten text recognition.

Supports:
- CodeMaster-CRNN (CodeMaster-AJ / IAM Word benchmark) as primary engine with word segmentation
- StandaloneLaiaCRNN (IAM Line benchmark)
- Legacy CRNN architecture fallback
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

from src.recognition.decoder import GreedyCTCDecoder
from src.recognition.model import CodeMasterCRNN, CRNNRecognitionModel, StandaloneLaiaCRNN
from src.recognition.preprocessing import LineImagePreprocessor
from src.recognition.vocabulary import Vocabulary

DEFAULT_CODEMASTER_ALPHABET = "!\"#'()*,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


class HandwritingPredictor:
    """Singleton-friendly inference service for handwriting recognition."""

    _instance: Optional["HandwritingPredictor"] = None

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "models/checkpoints/codemaster_crnn_trained.pth",
        device_name: str = "",
    ):
        """Initialize HandwritingPredictor.

        Args:
            checkpoint_path: Path to PyTorch checkpoint.
            device_name: 'cuda' or 'cpu' (auto-selected if empty).
        """
        ck_path = Path(checkpoint_path)

        # Candidate fallback order for checkpoints
        if not ck_path.is_file():
            candidates = [
                Path("models/checkpoints/codemaster_crnn_trained.pth"),
                Path("models/checkpoints/codemaster_crnn_finetuned.pth"),
                Path("models/checkpoints/codemaster_crnn_best.pth"),
                Path("models/checkpoints/pylaia_iam_crnn.pt"),
                Path("models/checkpoints/iam_crnn_best.pt"),
            ]
            for cand in candidates:
                if cand.is_file():
                    ck_path = cand
                    break

        self.checkpoint_path = ck_path
        self.device = torch.device(
            device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = None
        self.vocabulary: Optional[Vocabulary] = None
        self.decoder: Optional[GreedyCTCDecoder] = None
        self.preprocessor: Optional[LineImagePreprocessor] = None
        self.is_codemaster: bool = False
        self.is_pylaia: bool = False
        self.is_ready: bool = False
        self.status_message: str = ""

        # Mapping dictionaries for fast decoding
        self.char_to_num = {}
        self.num_to_char = {}

        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Load trained weights, architecture, and vocabulary from checkpoint file."""
        if not self.checkpoint_path.is_file():
            self.is_ready = False
            self.status_message = (
                f"Model checkpoint not found at '{self.checkpoint_path}'. "
                f"Please ensure weights exist in models/checkpoints/."
            )
            return

        try:
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)

            if "idx_to_sym" in ckpt:
                # Benchmark PyLaia IAM CRNN model
                idx_to_sym = ckpt["idx_to_sym"]
                self.vocabulary = Vocabulary(use_unk=True)
                self.vocabulary.char_to_idx = {str(sym): int(idx) for idx, sym in idx_to_sym.items()}
                self.vocabulary.idx_to_char = {int(idx): str(sym) for idx, sym in idx_to_sym.items()}
                self.vocabulary.unk_index = 79
                self.decoder = GreedyCTCDecoder(vocabulary=self.vocabulary, blank_index=0)
                self.model = StandaloneLaiaCRNN(num_classes=len(idx_to_sym)).to(self.device)
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.model.eval()
                self.is_codemaster = False
                self.is_pylaia = True
                self.is_ready = True
                self.status_message = (
                    f"PyLaia-CRNN active (IAM benchmark line model, {len(idx_to_sym)} classes)."
                )
                return

            # Check if this is a CodeMaster-CRNN checkpoint
            is_cm = (
                "codemaster" in self.checkpoint_path.name.lower()
                or (isinstance(ckpt, dict) and "linear.weight" in ckpt and "fc.weight" in ckpt)
                or (isinstance(ckpt, dict) and "model_state_dict" in ckpt and "fc.weight" in ckpt["model_state_dict"])
            )

            if is_cm:
                state_dict = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt
                fc_weight = state_dict.get("fc.weight", None)
                num_classes = fc_weight.shape[0] if fc_weight is not None else 77

                # Initialize alphabet and mappings
                alphabet = list(DEFAULT_CODEMASTER_ALPHABET)
                self.char_to_num = {char: idx + 1 for idx, char in enumerate(alphabet)}
                self.num_to_char = {idx + 1: char for idx, char in enumerate(alphabet)}

                self.vocabulary = Vocabulary(use_unk=False)
                self.vocabulary.char_to_idx = {char: idx + 1 for idx, char in enumerate(alphabet)}
                self.vocabulary.idx_to_char = {idx + 1: char for idx, char in enumerate(alphabet)}
                self.decoder = GreedyCTCDecoder(vocabulary=self.vocabulary, blank_index=0)

                self.model = CodeMasterCRNN(num_classes=num_classes).to(self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()

                self.is_codemaster = True
                self.is_pylaia = False
                self.is_ready = True
                self.status_message = (
                    f"CodeMaster-CRNN active ({self.checkpoint_path.name}, {num_classes} classes, PyTorch CNN+BiLSTM+CTC)."
                )
                return

            # Legacy CRNN checkpoint format
            vocab_dict = ckpt["vocabulary"]
            self.vocabulary = Vocabulary.from_dict(vocab_dict)
            self.decoder = GreedyCTCDecoder(
                vocabulary=self.vocabulary, blank_index=Vocabulary.BLANK_INDEX
            )

            prep_cfg = ckpt.get("preprocessing_config", {})
            self.preprocessor = LineImagePreprocessor(
                target_height=prep_cfg.get("target_height", 32),
                min_width=prep_cfg.get("min_width", 32),
                invert_colors=prep_cfg.get("invert_colors", True),
            )

            model_cfg = ckpt.get("model_config", {})
            self.model = CRNNRecognitionModel(
                num_classes=len(self.vocabulary),
                in_channels=model_cfg.get("in_channels", 1),
                lstm_hidden_size=model_cfg.get("lstm_hidden_size", 256),
            ).to(self.device)

            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()

            self.is_codemaster = False
            self.is_pylaia = False
            self.is_ready = True
            val_cer = ckpt.get("val_cer", None)
            cer_info = f" (Val CER: {val_cer:.2%})" if val_cer is not None else ""
            self.status_message = f"CRNN model loaded successfully from '{self.checkpoint_path.name}'{cer_info}."
        except Exception as e:
            self.is_ready = False
            self.status_message = f"Failed to load checkpoint '{self.checkpoint_path}': {e}"

    def recognize_word_crop(self, word_crop: np.ndarray) -> Tuple[str, float]:
        """Recognize a single word image using CodeMaster-CRNN.

        Args:
            word_crop: Grayscale or BGR cropped word image.

        Returns:
            Tuple[str, float]: (recognized_word, confidence_score).
        """
        if not self.is_ready or self.model is None:
            return "", 0.0

        if word_crop.ndim == 3:
            gray = cv2.cvtColor(word_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = word_crop

        h, w = gray.shape[:2]
        if h < 4 or w < 4:
            return "", 0.0

        # Discard blank crops
        if np.std(gray) < 5:
            return "", 0.0

        # Fit aspect ratio inside (32, 128) canvas
        f = min(128.0 / max(1, w), 32.0 / max(1, h))
        new_w, new_h = max(1, int(w * f)), max(1, int(h * f))
        img_res = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.ones((32, 128), dtype=np.uint8) * 255
        canvas[0:new_h, 0:new_w] = img_res

        # Ensure standard black ink on white paper (match IAM training distribution)
        if np.median(gray) < 100:
            # Document has dark background with light ink: invert it
            canvas = 255 - canvas

        tensor_input = (
            torch.tensor(canvas.astype(np.float32) / 255.0, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        with torch.no_grad():
            preds = self.model(tensor_input)  # (1, 32, num_classes)
            probs = torch.softmax(preds, dim=2).squeeze(0)  # (32, num_classes)
            pred_indexes = torch.argmax(probs, dim=1).cpu().numpy()

        # CTC Greedy Decode
        decoded_string = ""
        last_seen_token = None
        conf_scores = []
        for t_step, token in enumerate(pred_indexes):
            if token != 0 and token != last_seen_token:
                char = self.num_to_char.get(token, "")
                decoded_string += char
                conf_scores.append(float(probs[t_step, token].item()))
            last_seen_token = token

        conf = float(np.mean(conf_scores)) if conf_scores else 0.0
        return decoded_string, conf

    def segment_line_words(self, line_gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Segment a horizontal line crop into individual word bounding boxes using projection profiling."""
        from src.preprocessing.word_segmenter import WordSegmenter
        return WordSegmenter().segment_words(line_gray)

    def recognize_line(
        self, image: Union[Image.Image, np.ndarray]
    ) -> Tuple[str, float]:
        """Recognize handwritten text in a line image strip.

        For CodeMaster-CRNN:
          Splits multi-word lines into individual word crops,
          transcribes each word crop, and joins them with whitespace.

        Args:
            image: Cropped line image (PIL Image or NumPy array).

        Returns:
            Tuple[str, float]: (recognized_text, confidence_score).
        """
        if not self.is_ready or self.model is None or self.decoder is None:
            return "", 0.0

        if isinstance(image, Image.Image):
            arr = np.array(image)
        else:
            arr = image.copy()

        if arr.ndim == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        else:
            gray = arr

        h, w = gray.shape[:2]
        if h < 4 or w < 8:
            return "", 0.0

        # -------------------------------------------------------------
        # 1. CodeMaster-CRNN Path (Word-level engine with line slicing)
        # -------------------------------------------------------------
        if self.is_codemaster:
            # If the crop is a multi-word line or wide phrase
            if (w > 70 and (w / max(1, h)) > 1.8) or w > 110:
                word_boxes = self.segment_line_words(gray)
                if len(word_boxes) > 1:
                    words = []
                    confs = []
                    for wb in word_boxes:
                        wx, wy, ww, wh = wb
                        wc = gray[wy : wy + wh, wx : wx + ww]
                        w_text, w_conf = self.recognize_word_crop(wc)
                        if w_text:
                            words.append(w_text)
                            confs.append(w_conf)
                    if words:
                        return " ".join(words), float(np.mean(confs)) if confs else 0.0

            # Single word crop or fallback
            return self.recognize_word_crop(gray)

        # -------------------------------------------------------------
        # 2. PyLaia Line Engine Path
        # -------------------------------------------------------------
        if self.is_pylaia:
            bg = cv2.medianBlur(cv2.dilate(gray, np.ones((5, 5), np.uint8)), 21)
            diff = 255 - cv2.absdiff(gray, bg)
            norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            new_w = max(32, int(w * (128.0 / max(1, h))))
            if new_w % 8 != 0:
                new_w = ((new_w // 8) + 1) * 8
            resized = cv2.resize(norm, (new_w, 128), interpolation=cv2.INTER_AREA)

            tensor_np = 1.0 - (resized.astype(np.float32) / 255.0)
            tensor = torch.tensor(tensor_np, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                log_probs = self.model.compute_log_probs(tensor)
                results = self.decoder.decode(log_probs)

            if results:
                text, conf = results[0]
                return text.strip(), conf
            return "", 0.0

        # -------------------------------------------------------------
        # 3. Legacy Model Path
        # -------------------------------------------------------------
        if self.preprocessor is not None:
            tensor = self.preprocessor.preprocess(arr, augment=False)
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            tensor = tensor.to(self.device)
            input_len = torch.tensor([max(1, (tensor.shape[3] // 4) - 1)], dtype=torch.long, device=self.device)

            with torch.no_grad():
                log_probs = self.model.compute_log_probs(tensor)
                results = self.decoder.decode(log_probs, input_lengths=input_len)

            if results:
                text, conf = results[0]
                return text.strip(), conf

        return "", 0.0


# Module-level convenience functions
_DEFAULT_PREDICTOR: Optional[HandwritingPredictor] = None


def get_predictor(
    checkpoint_path: str = "models/checkpoints/codemaster_crnn_trained.pth",
) -> HandwritingPredictor:
    """Return a cached singleton HandwritingPredictor instance."""
    global _DEFAULT_PREDICTOR
    if _DEFAULT_PREDICTOR is None or str(_DEFAULT_PREDICTOR.checkpoint_path) != checkpoint_path:
        _DEFAULT_PREDICTOR = HandwritingPredictor(checkpoint_path=checkpoint_path)
    return _DEFAULT_PREDICTOR


def recognize_line(image: Union[Image.Image, np.ndarray]) -> str:
    """High-level function: recognize_line(image) -> recognized_text."""
    predictor = get_predictor()
    text, _ = predictor.recognize_line(image)
    return text
