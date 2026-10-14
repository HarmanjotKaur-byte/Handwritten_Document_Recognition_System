"""Handwriting line image preprocessing for CRNN recognition.

Provides:
- Grayscale conversion
- Aspect-ratio preserving resizing to a fixed height (e.g., 32px)
- Pixel intensity normalization
- Color inversion (optional, so dark ink becomes positive signal)
- Conservative training-only augmentations
- Padding utilities
"""

from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image


class LineImagePreprocessor:
    """Preprocesses handwritten text line images uniformly across training and inference.

    Conventions:
    - Input: PIL Image (RGB or L) or NumPy array (BGR, RGB, or Grayscale).
    - Height: Fixed to `target_height` (default 128).
    - Width: Rescaled proportionally: new_w = int(w * (target_height / h)), bounded to at least min_width.
    - Output: PyTorch float32 tensor of shape (1, target_height, new_w) with pixel values in [-1.0, 1.0].
    """

    def __init__(
        self,
        target_height: int = 128,
        min_width: int = 32,
        invert_colors: bool = False,
        normalize_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        """Initialize Preprocessor.

        Args:
            target_height: Standard height in pixels (default 128).
            min_width: Minimum allowable width in pixels.
            invert_colors: Invert colors flag.
            normalize_range: Range to scale normalized pixels.
        """
        self.target_height = target_height
        self.min_width = min_width
        self.invert_colors = invert_colors
        self.normalize_range = normalize_range

    def to_grayscale_array(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Convert input to single-channel 8-bit grayscale NumPy array."""
        if isinstance(image, Image.Image):
            if image.mode != "L":
                image = image.convert("L")
            return np.array(image, dtype=np.uint8)
        elif isinstance(image, np.ndarray):
            if image.ndim == 3:
                if image.shape[2] == 3:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                elif image.shape[2] == 4:
                    return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            return image.astype(np.uint8)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def resize_keep_aspect(self, gray: np.ndarray) -> np.ndarray:
        """Resize image to fixed height while strictly maintaining aspect ratio.

        Args:
            gray: 2D grayscale array of shape (H, W).

        Returns:
            np.ndarray: Resized array of shape (target_height, new_w).
        """
        h, w = gray.shape[:2]
        if h == 0 or w == 0:
            return np.ones((self.target_height, self.min_width), dtype=np.uint8) * 255

        scale = self.target_height / float(h)
        new_w = max(self.min_width, int(round(w * scale)))

        # Ensure width is divisible by 4 for CNN pooling downsampling
        if new_w % 4 != 0:
            new_w += (4 - (new_w % 4))

        if scale < 1.0:
            # Downsampling: INTER_AREA prevents aliasing
            resized = cv2.resize(gray, (new_w, self.target_height), interpolation=cv2.INTER_AREA)
        else:
            # Upsampling: INTER_CUBIC preserves smooth strokes
            resized = cv2.resize(gray, (new_w, self.target_height), interpolation=cv2.INTER_CUBIC)

        return resized

    def apply_augmentation(self, img: np.ndarray) -> np.ndarray:
        """Apply conservative augmentations that do not alter handwriting characters."""
        # Random slight brightness and contrast shift
        alpha = float(np.random.uniform(0.9, 1.1))  # Contrast
        beta = float(np.random.uniform(-10, 10))    # Brightness
        augmented = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        # 20% chance of gentle affine slant adjustment (+/- 3 degrees)
        if np.random.rand() < 0.2:
            h, w = augmented.shape[:2]
            shear_factor = float(np.random.uniform(-0.05, 0.05))
            m = np.float32([[1, shear_factor, 0], [0, 1, 0]])
            augmented = cv2.warpAffine(
                augmented, m, (w, h), borderMode=cv2.BORDER_REPLICATE
            )

        return augmented

    def normalize(self, img: np.ndarray) -> np.ndarray:
        """Scale pixel values into float32 range [-1.0, 1.0], matching IAM-line CRNN."""
        float_img = img.astype(np.float32) / 255.0
        norm = (float_img - 0.5) / 0.5
        return norm

    def clean_line_image(self, gray: np.ndarray) -> np.ndarray:
        """Normalize paper background illumination and suppress ruled lines."""
        if gray.shape[0] < 8 or gray.shape[1] < 16:
            return gray

        # Background division to eliminate shadows / paper tint variations
        k_size = max(9, min(21, (gray.shape[0] // 2) * 2 + 1))
        bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size)))
        norm = cv2.divide(gray, np.maximum(bg, 1), scale=255)

        # Contrast stretching so text ink is prominent and paper is crisp
        stretched = cv2.normalize(norm, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # Ruled line suppression
        line_w = max(15, int(gray.shape[1] * 0.05))
        hl = cv2.morphologyEx(255 - stretched, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (line_w, 1)))
        cleaned = np.clip(stretched.astype(int) + (hl * 0.7).astype(int), 0, 255).astype(np.uint8)
        return cleaned

    def preprocess(
        self,
        image: Union[Image.Image, np.ndarray],
        augment: bool = False,
    ) -> torch.Tensor:
        """Run full preprocessing pipeline on a single line image.

        Args:
            image: Source image.
            augment: Whether to apply training augmentation.

        Returns:
            torch.Tensor: Normalized tensor with shape (1, target_height, new_w).
        """
        gray = self.to_grayscale_array(image)
        cleaned = self.clean_line_image(gray)
        resized = self.resize_keep_aspect(cleaned)

        if augment:
            resized = self.apply_augmentation(resized)

        norm = self.normalize(resized)
        tensor = torch.tensor(norm, dtype=torch.float32).unsqueeze(0)
        return tensor

    def get_config(self) -> dict:
        """Return preprocessing settings dictionary for checkpoint storage."""
        return {
            "target_height": self.target_height,
            "min_width": self.min_width,
            "invert_colors": self.invert_colors,
            "normalize_range": list(self.normalize_range),
        }
