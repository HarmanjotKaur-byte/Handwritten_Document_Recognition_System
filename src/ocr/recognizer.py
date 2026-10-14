"""High-level handwriting recognition inference pipeline.

Encapsulates:
- TrOCR (Vision Transformer) for state-of-the-art 95%+ accuracy
- CodeMaster-CRNN and PyLaia-CRNN for fast lightweight CPU inference
- Line batching, device management, and CTC greedy decoding
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from src.config import OCR_MODEL_PATH, get_torch_device
from src.ocr.decoder import CTCDecoder
from src.ocr.model import CRNN
from src.preprocessing.image_processor import ImageProcessor
from src.preprocessing.segmenter import LineSegment


@dataclass
class WordRecognitionResult:
    """Recognition result for an individual word inside a line."""

    word_index: int
    text: str
    confidence: float
    bbox: tuple  # (x, y, w, h) in absolute page coordinates


@dataclass
class LineRecognitionResult:
    """Recognition result for a single segmented line."""

    line_index: int
    text: str
    confidence: float
    bbox: tuple  # (x, y, w, h)
    words: List[WordRecognitionResult] = field(default_factory=list)


@dataclass
class OCRResult:
    """Aggregated recognition result across all document lines."""

    full_text: str
    lines: List[LineRecognitionResult] = field(default_factory=list)
    model_loaded: bool = False
    is_trained: bool = False
    status_message: str = ""


class HandwritingRecognizer:
    """Manages PyTorch CRNN and TrOCR model lifecycle, weights, and line inference."""

    def __init__(
        self,
        model_path: Optional[Path] = OCR_MODEL_PATH,
        engine_type: str = "trocr",
    ):
        """Initialize the recognizer.

        Args:
            model_path (Optional[Path]): Path to trained .pth weights file.
            engine_type (str): 'trocr', 'codemaster', or 'pylaia'.
        """
        self.model_path = Path(model_path) if model_path else None
        self.device = torch.device(get_torch_device())
        self.decoder = CTCDecoder()
        self.processor = ImageProcessor()
        self.engine_type = engine_type

        # Initialize network architecture
        self.model = CRNN(num_classes=self.decoder.num_classes)
        self.model.to(self.device)
        self.model.eval()

        self.model_loaded: bool = False
        self.is_trained: bool = False
        self.iam_predictor = None
        self.trocr_engine = None

        self._load_weights_if_available()

    def _load_weights_if_available(self) -> None:
        """Attempt loading trained PyTorch checkpoint weights if present."""
        # Check TrOCR first if requested
        if self.engine_type == "trocr":
            try:
                from src.recognition.trocr_engine import get_trocr_engine

                self.trocr_engine = get_trocr_engine()
                if self.trocr_engine.is_ready:
                    self.model_loaded = True
                    self.is_trained = True
            except Exception:
                self.trocr_engine = None

        # Load CRNN predictor matching requested engine_type
        if self.engine_type == "pylaia":
            candidates = [
                Path("models/checkpoints/pylaia_iam_crnn.pt"),
                Path("models/checkpoints/iam_crnn_best.pt"),
            ]
        else:
            candidates = [
                Path("models/checkpoints/codemaster_crnn_trained.pth"),
                Path("models/checkpoints/codemaster_crnn_finetuned.pth"),
                Path("models/checkpoints/codemaster_crnn_best.pth"),
                Path("models/checkpoints/pylaia_iam_crnn.pt"),
            ]
        target_ckpt = None
        for cand in candidates:
            if cand.is_file():
                target_ckpt = cand
                break

        if target_ckpt is not None:
            try:
                from src.recognition.predictor import HandwritingPredictor

                self.iam_predictor = HandwritingPredictor(checkpoint_path=target_ckpt)
                if self.iam_predictor.is_ready:
                    self.model_loaded = True
                    self.is_trained = True
                    self.model_path = target_ckpt
            except Exception:
                self.iam_predictor = None

        if self.trocr_engine is not None and self.trocr_engine.is_ready:
            self.model_loaded = True
            self.is_trained = True
        elif self.iam_predictor is not None and self.iam_predictor.is_ready:
            self.model_loaded = True
            self.is_trained = True

    def ensure_engine_loaded(self, engine_type: str) -> None:
        """Ensure the specified engine is loaded into memory."""
        self.engine_type = engine_type
        if engine_type == "trocr":
            if self.trocr_engine is None or not self.trocr_engine.is_ready:
                try:
                    from src.recognition.trocr_engine import get_trocr_engine
                    self.trocr_engine = get_trocr_engine()
                    if self.trocr_engine.is_ready:
                        self.model_loaded = True
                        self.is_trained = True
                except Exception as e:
                    print(f"Failed to load TrOCR: {e}")
                    self.trocr_engine = None
        else:
            if self.iam_predictor is None or not self.iam_predictor.is_ready:
                self._load_weights_if_available()

    def get_status_diagnostics(self) -> dict:
        """Return diagnostic information regarding device and model state."""
        self.ensure_engine_loaded(self.engine_type)
        active_engine = (
            "TrOCR (Vision Transformer)"
            if (self.engine_type == "trocr" and self.trocr_engine and self.trocr_engine.is_ready)
            else (
                "CodeMaster-CRNN"
                if (self.iam_predictor and self.iam_predictor.is_codemaster)
                else "CRNN Baseline"
            )
        )
        return {
            "device": str(self.device),
            "engine": active_engine,
            "engine_type": self.engine_type,
            "model_path": str(self.model_path) if self.model_path else "None",
            "weights_found": True if (self.is_trained or (self.trocr_engine and self.trocr_engine.is_ready)) else False,
            "is_trained": bool(self.is_trained or (self.trocr_engine and self.trocr_engine.is_ready)),
            "trocr_ready": bool(self.trocr_engine and self.trocr_engine.is_ready),
        }

    def recognize_lines(
        self, segments: List[LineSegment], apply_lexicon: bool = True
    ) -> OCRResult:
        """Run inference over a list of line segments.

        Args:
            segments (List[LineSegment]): Detected line strips.
            apply_lexicon (bool): Whether to apply dictionary language model post-processing.

        Returns:
            OCRResult: Decoded line texts, confidence scores, and aggregate full text.
        """
        self.ensure_engine_loaded(self.engine_type)
        if not segments:
            return OCRResult(
                full_text="",
                lines=[],
                model_loaded=self.model_loaded,
                is_trained=self.is_trained,
                status_message="No text lines were detected during document segmentation.",
            )

        line_results: List[LineRecognitionResult] = []
        corrector = None
        if apply_lexicon:
            try:
                from src.recognition.lexicon_corrector import get_lexicon_corrector
                corrector = get_lexicon_corrector()
            except Exception:
                corrector = None

        with torch.no_grad():
            for seg in segments:
                decoded_text = ""
                conf = 0.0

                # 1. TrOCR High-Accuracy Path
                if (
                    self.engine_type == "trocr"
                    and self.trocr_engine is not None
                    and self.trocr_engine.is_ready
                ):
                    decoded_text, conf = self.trocr_engine.recognize_line(seg.crop)

                # 2. PyTorch CRNN (CodeMaster / PyLaia) Path
                elif self.iam_predictor is not None and self.iam_predictor.is_ready:
                    decoded_text, conf = self.iam_predictor.recognize_line(seg.crop)

                # 3. Baseline Fallback
                else:
                    norm_img = self.processor.normalize_line_image(seg.crop)
                    tensor_img = (
                        torch.tensor(norm_img, dtype=torch.float32)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    w = tensor_img.shape[3]
                    if w % 4 != 0:
                        pad_w = 4 - (w % 4)
                        pad_tensor = torch.ones((1, 1, 32, pad_w), device=self.device)
                        tensor_img = torch.cat([tensor_img, pad_tensor], dim=3)

                    logits = self.model(tensor_img)
                    texts, confs = self.decoder.decode_greedy(logits)
                    decoded_text = texts[0] if texts else ""
                    conf = confs[0] if confs else 0.0

                raw_text = decoded_text.strip()
                # Clean stray punctuation prefixes from crop boundaries (#, -, |, *)
                cleaned_line = re.sub(r"^[#\-\*\.\,\:\;\|\s\(\)]+", "", raw_text).strip()
                cleaned_line = re.sub(r"[\s\(\)]+$", "", cleaned_line).strip()

                # Filter out non-text noise (e.g. notebook headers '0 0' or pen clips '000 (')
                has_alpha = bool(re.search(r"[a-zA-Z]", cleaned_line))
                if not has_alpha:
                    continue

                if corrector is not None and cleaned_line:
                    final_text = corrector.correct_sentence(cleaned_line)
                else:
                    final_text = cleaned_line

                # Extract individual word bounding boxes along the line
                line_words: List[WordRecognitionResult] = []
                tokens = final_text.split()
                if tokens:
                    try:
                        from src.preprocessing.word_segmenter import WordSegmenter
                        w_boxes = WordSegmenter().segment_words(seg.crop)
                    except Exception:
                        w_boxes = []

                    num_b = len(w_boxes)
                    num_t = len(tokens)
                    if num_b == num_t:
                        for wi, (tok, wb) in enumerate(zip(tokens, w_boxes)):
                            wx, wy, ww, wh = wb
                            abs_bbox = (seg.x + wx, seg.y + wy, ww, wh)
                            line_words.append(WordRecognitionResult(wi, tok, conf, abs_bbox))
                    elif num_b > 0:
                        for wi, tok in enumerate(tokens):
                            bi = min(num_b - 1, int(round(wi * (num_b - 1) / max(1, num_t - 1))))
                            wx, wy, ww, wh = w_boxes[bi]
                            abs_bbox = (seg.x + wx, seg.y + wy, ww, wh)
                            line_words.append(WordRecognitionResult(wi, tok, conf, abs_bbox))
                    else:
                        line_w = seg.w
                        curr_x = seg.x
                        for wi, tok in enumerate(tokens):
                            est_w = max(18, int(len(tok) * (line_w / max(1, len(final_text)))))
                            abs_bbox = (curr_x, seg.y, est_w, seg.h)
                            line_words.append(WordRecognitionResult(wi, tok, conf, abs_bbox))
                            curr_x += est_w + 8

                line_results.append(
                    LineRecognitionResult(
                        line_index=seg.index,
                        text=final_text,
                        confidence=conf,
                        bbox=(seg.x, seg.y, seg.w, seg.h),
                        words=line_words,
                    )
                )

        full_text = "\n".join(r.text for r in line_results if r.text).strip()

        status_msg = (
            f"Inference successful using {self.get_status_diagnostics()['engine']}."
            if self.is_trained
            else "Model is currently UNTRAINED."
        )

        return OCRResult(
            full_text=full_text,
            lines=line_results,
            model_loaded=self.model_loaded,
            is_trained=self.is_trained,
            status_message=status_msg,
        )
