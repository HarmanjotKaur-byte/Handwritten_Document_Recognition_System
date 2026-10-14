"""TrOCR (Vision Transformer) Inference Engine for State-of-the-Art Handwriting Recognition.

Uses Microsoft's official VisionEncoderDecoderModel (microsoft/trocr-base-handwritten).
Employs standard Hugging Face pipeline without manual distortion or canvas padding.
"""

from typing import Optional, Tuple, Union
import cv2
import numpy as np
import torch
from PIL import Image


class TrOCRInferenceEngine:
    """Manages TrOCR Processor and VisionEncoderDecoderModel for handwriting recognition."""

    _instance: Optional["TrOCRInferenceEngine"] = None

    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-handwritten",
        device_name: str = "",
    ):
        """Initialize TrOCR engine.

        Args:
            model_name: HuggingFace model identifier.
            device_name: 'cuda' or 'cpu'.
        """
        self.model_name = model_name
        self.device = torch.device(
            device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.processor = None
        self.model = None
        self.is_ready = False
        self.status_message = "TrOCR engine initializing..."

        self._load_model()

    def _load_model(self) -> None:
        """Load TrOCR processor and model weights."""
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            print(f"[TrOCR] Loading official pretrained model '{self.model_name}' on {self.device}...")
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.is_ready = True
            self.status_message = f"TrOCR active ({self.model_name}, SOTA Vision Transformer on {self.device})."
            print(f"[TrOCR] Model loaded successfully on {self.device}.")
        except Exception as e:
            self.is_ready = False
            self.status_message = f"Failed to load TrOCR ({e})."
            print(f"[TrOCR] Failed to load model: {e}")

    def recognize_line(
        self, image: Union[Image.Image, np.ndarray], max_new_tokens: int = 64
    ) -> Tuple[str, float]:
        """Recognize handwritten text in a line image strip using TrOCR.

        Args:
            image: Line crop (NumPy array or PIL Image).
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Tuple[str, float]: (recognized_text, confidence_score).
        """
        if not self.is_ready or self.model is None or self.processor is None:
            return "", 0.0

        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            arr = image.copy()
            if arr.ndim == 2:
                rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif arr.shape[2] == 4:
                rgb = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            else:
                rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

        w, h = pil_image.size
        if h < 6 or w < 12:
            return "", 0.0

        # High-resolution scaling guard: on full-page mobile photos, line strips can be 15-25px tall.
        # When TrOCR's internal ViT processor resizes 15px lines into 384x384, ink strokes get blurred.
        # Proportional bicubic upsampling to at least 40px preserves stroke fidelity.
        min_target_h = 40
        if h < min_target_h:
            scale_factor = float(min_target_h) / float(h)
            new_w = max(16, int(round(w * scale_factor)))
            pil_image = pil_image.resize((new_w, min_target_h), Image.Resampling.BICUBIC)

        # Discard blank strips
        gray = np.array(pil_image.convert("L"))
        if np.std(gray) < 6:
            return "", 0.0

        try:
            # Official expected preprocessing: processor resizes and normalizes directly
            pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            with torch.no_grad():
                generated_outputs = self.model.generate(
                    pixel_values,
                    max_new_tokens=max_new_tokens,
                    num_beams=4,
                    early_stopping=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            generated_ids = generated_outputs.sequences
            decoded_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            confidence = 0.95
            if hasattr(generated_outputs, "sequences_scores") and generated_outputs.sequences_scores is not None:
                log_prob = generated_outputs.sequences_scores[0].item()
                seq_len = max(1, len(generated_ids[0]))
                confidence = float(np.clip(np.exp(log_prob / seq_len), 0.1, 1.0))

            return decoded_text, confidence

        except Exception as e:
            print(f"[TrOCR] Inference error: {e}")
            return "", 0.0


# Singleton accessor
_GLOBAL_TROCR_ENGINE: Optional[TrOCRInferenceEngine] = None


def get_trocr_engine(model_name: str = "microsoft/trocr-base-handwritten") -> TrOCRInferenceEngine:
    """Return a cached singleton instance of TrOCRInferenceEngine."""
    global _GLOBAL_TROCR_ENGINE
    if _GLOBAL_TROCR_ENGINE is None:
        _GLOBAL_TROCR_ENGINE = TrOCRInferenceEngine(model_name=model_name)
    return _GLOBAL_TROCR_ENGINE
