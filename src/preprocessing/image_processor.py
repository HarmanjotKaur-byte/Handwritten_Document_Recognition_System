"""Handwriting-preserving image preprocessing pipeline using OpenCV, Pillow, and NumPy.

Includes grayscale conversion, edge-preserving bilateral denoising,
Contrast Limited Adaptive Histogram Equalization (CLAHE), optional binarization
(Otsu, Adaptive Gaussian, Sauvola, or None), deskewing, and aspect-ratio preserving resizing.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from src.config import OCR_IMAGE_HEIGHT, OCR_IMAGE_MIN_WIDTH


@dataclass
class PreprocessingOptions:
    """Configurable options for document image preprocessing."""

    apply_flatten_illumination: bool = True  # Flattens shadows, vignettes, and uneven lighting
    apply_grayscale: bool = True
    apply_denoise: bool = True
    denoise_method: str = "bilateral"  # "bilateral" or "gaussian"
    apply_contrast_enhancement: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    apply_deskew: bool = True
    max_deskew_angle: float = 15.0  # Guard against rotating cursive handwriting slant
    binarize_method: str = "none"  # "none", "otsu", "adaptive", "sauvola"
    max_dimension: Optional[int] = 1800  # Max width/height to constrain memory and latency


@dataclass
class PreprocessedDocument:
    """Holds intermediate and final representations of a preprocessed page."""

    original: np.ndarray             # Original BGR image array
    grayscale: np.ndarray            # Grayscale image
    denoised: np.ndarray             # Denoised grayscale image
    contrast_enhanced: np.ndarray    # CLAHE contrast-enhanced image
    deskewed: np.ndarray             # Rotated deskewed image
    processed: np.ndarray            # Final output image ready for inspection/downstream
    is_binarized: bool               # Whether output is thresholded black-and-white
    skew_angle: float = 0.0          # Estimated skew angle in degrees
    original_size: Tuple[int, int] = (0, 0)   # (width, height)
    processed_size: Tuple[int, int] = (0, 0)  # (width, height)


class ImageProcessor:
    """Preprocesses handwritten document images while preserving delicate ink strokes."""

    def __init__(self, target_line_height: int = OCR_IMAGE_HEIGHT):
        """Initialize ImageProcessor.

        Args:
            target_line_height (int): Standard height in pixels for line crops.
        """
        self.target_line_height = target_line_height

    def to_cv2(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Convert PIL Image or array to OpenCV BGR numpy array.

        Args:
            image (Union[Image.Image, np.ndarray]): Source image.

        Returns:
            np.ndarray: BGR image array (uint8).
        """
        if isinstance(image, Image.Image):
            rgb = np.array(image.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.ndim == 3 and image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            return image.copy()
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def resize_with_aspect_ratio(
        self, image: np.ndarray, max_dim: Optional[int] = 1800
    ) -> np.ndarray:
        """Resize image to stay within max_dimension while strictly preserving aspect ratio.

        Avoids aggressive downsampling that could thin out faint handwriting strokes.

        Args:
            image (np.ndarray): Image array.
            max_dim (Optional[int]): Maximum width or height constraint.

        Returns:
            np.ndarray: Scaled image array.
        """
        if max_dim is None or max_dim <= 0:
            return image

        h, w = image.shape[:2]
        if max(h, w) <= max_dim:
            return image

        if h > w:
            new_h = max_dim
            new_w = int(w * (max_dim / float(h)))
        else:
            new_w = max_dim
            new_h = int(h * (max_dim / float(w)))

        # cv2.INTER_AREA is optimal for decimation/downsampling without introducing aliasing artifacts
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR image to single-channel 8-bit grayscale.

        Args:
            image (np.ndarray): BGR image.

        Returns:
            np.ndarray: Grayscale image.
        """
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def flatten_background_illumination(self, gray: np.ndarray) -> np.ndarray:
        """Flatten non-uniform shadows, vignette gradients, and phone camera lighting.

        Divides the grayscale image by a morphological background estimate to produce
        uniform paper white across the whole page, preventing dark thresholding blobs.

        Args:
            gray (np.ndarray): Single-channel grayscale image.

        Returns:
            np.ndarray: Illumination-normalized grayscale image.
        """
        # Estimate the background surface using morphological dilation followed by median blur
        kernel_size = 21
        dilated = cv2.dilate(
            gray,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)),
        )
        bg = cv2.medianBlur(dilated, 25)

        # Division normalization: gray / bg * 255
        diff = cv2.divide(gray, bg, scale=255)
        # Normalize contrast to full 0..255 dynamic range
        normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        return normalized

    def denoise(self, gray: np.ndarray, method: str = "bilateral") -> np.ndarray:
        """Apply edge-preserving noise reduction without smoothing away handwriting strokes.

        Bilateral filtering is preferred over median or Gaussian blur because it preserves
        sharp boundary transitions of pen and pencil strokes while eliminating paper grain.

        Args:
            gray (np.ndarray): Grayscale image.
            method (str): 'bilateral' or 'gaussian'.

        Returns:
            np.ndarray: Denoised grayscale image.
        """
        if method == "gaussian":
            # Gentle 3x3 Gaussian smoothing
            return cv2.GaussianBlur(gray, (3, 3), 0)

        # Gentle bilateral filter: d=5 (diameter), sigmaColor=50, sigmaSpace=50
        return cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    def enhance_contrast(
        self,
        gray: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """Enhance local contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Brings out faint pencil marks and faded ballpoint ink without blowing out background paper.

        Args:
            gray (np.ndarray): Grayscale image.
            clip_limit (float): Threshold for contrast limiting.
            tile_grid_size (Tuple[int, int]): Size of grid for histogram equalization.

        Returns:
            np.ndarray: Contrast-enhanced grayscale image.
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(gray)

    def detect_skew_angle(self, gray: np.ndarray) -> float:
        """Estimate the rotational skew angle of the page.

        Calculated via Otsu thresholding and minimum area bounding rectangle.
        Constrained to prevent confusing handwriting slant with page rotation.

        Args:
            gray (np.ndarray): Grayscale image.

        Returns:
            float: Estimated skew angle in degrees.
        """
        # First attempt: Hough line transform to detect paper ruled lines or text baselines
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        hough_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=70, minLineLength=60, maxLineGap=10)
        line_angles = []
        if hough_lines is not None:
            for l in hough_lines:
                x1, y1, x2, y2 = l.flatten()[:4]
                deg = float(np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1))))
                if abs(deg) < 20:
                    line_angles.append(deg)

        if len(line_angles) >= 3:
            median_angle = float(np.median(line_angles))
            if abs(median_angle) <= 15:
                return median_angle

        # Fallback: minAreaRect on binary text mask
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(thresh > 0))

        if len(coords) < 100:
            return 0.0

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) > 15:
            return 0.0

        return float(angle)

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by the given angle around its center with clean white border padding.

        Args:
            image (np.ndarray): Image array.
            angle (float): Rotation angle in degrees.

        Returns:
            np.ndarray: Deskewed image.
        """
        if abs(angle) < 0.2:
            return image

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        border_val = (255, 255, 255) if image.ndim == 3 else 255
        return cv2.warpAffine(
            image,
            rot_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_val,
        )

    def binarize(self, gray: np.ndarray, method: str = "none") -> Tuple[np.ndarray, bool]:
        """Optionally binarize grayscale image to high-contrast black and white.

        Supported methods:
        - 'none': Keep enhanced grayscale (recommended for faint handwriting preservation)
        - 'otsu': Global optimal variance thresholding
        - 'adaptive': Local adaptive Gaussian thresholding
        - 'sauvola': Sauvola local thresholding tailored for degraded handwriting documents

        Args:
            gray (np.ndarray): Grayscale image.
            method (str): Binarization strategy.

        Returns:
            Tuple[np.ndarray, bool]: (processed_image, is_binarized).
        """
        method_clean = (method or "none").lower()

        if method_clean in {"none", "grayscale", "raw"}:
            return gray.copy(), False

        elif method_clean == "otsu":
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary, True

        elif method_clean == "adaptive":
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 11
            )
            return binary, True

        elif method_clean == "sauvola":
            # Sauvola's thresholding formula: T = m * (1 + k * (s / R - 1))
            k = 0.2
            r = 128
            window_size = 25

            mean = cv2.boxFilter(gray.astype(np.float32), cv2.CV_32F, (window_size, window_size))
            sq_mean = cv2.boxFilter(
                (gray.astype(np.float32) ** 2), cv2.CV_32F, (window_size, window_size)
            )
            variance = np.maximum(0, sq_mean - (mean ** 2))
            std = np.sqrt(variance)

            threshold = mean * (1.0 + k * ((std / float(r)) - 1.0))
            binary = np.where(gray >= threshold, 255, 0).astype(np.uint8)
            return binary, True

        else:
            raise ValueError(f"Unknown binarization method: '{method}'. Options: none, otsu, adaptive, sauvola.")

    def process_page(
        self,
        image: Union[Image.Image, np.ndarray],
        options: Optional[PreprocessingOptions] = None,
    ) -> PreprocessedDocument:
        """Execute the full handwriting-safe preprocessing pipeline on a document page.

        Args:
            image (Union[Image.Image, np.ndarray]): Source image or page.
            options (Optional[PreprocessingOptions]): Pipeline configuration options.

        Returns:
            PreprocessedDocument: Comprehensive container with all intermediate stages.
        """
        if options is None:
            options = PreprocessingOptions()

        bgr_raw = self.to_cv2(image)
        orig_h, orig_w = bgr_raw.shape[:2]

        # 1. Aspect-ratio preserving resizing
        bgr = self.resize_with_aspect_ratio(bgr_raw, max_dim=options.max_dimension)
        proc_h, proc_w = bgr.shape[:2]

        # 2. Grayscale conversion
        if options.apply_grayscale:
            gray = self.to_grayscale(bgr)
        else:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 2.5. Shadow and Illumination Normalization
        if options.apply_flatten_illumination:
            flat_gray = self.flatten_background_illumination(gray)
        else:
            flat_gray = gray.copy()

        # 3. Edge-preserving Denoising
        if options.apply_denoise:
            denoised = self.denoise(flat_gray, method=options.denoise_method)
        else:
            denoised = flat_gray.copy()

        # 4. Contrast Enhancement (CLAHE)
        if options.apply_contrast_enhancement:
            enhanced = self.enhance_contrast(
                denoised,
                clip_limit=options.clahe_clip_limit,
                tile_grid_size=options.clahe_tile_grid_size,
            )
        else:
            enhanced = denoised.copy()

        # 5. Skew Detection & Safe Deskewing
        skew_angle = 0.0
        if options.apply_deskew:
            detected_angle = self.detect_skew_angle(enhanced)
            if abs(detected_angle) <= options.max_deskew_angle:
                skew_angle = detected_angle
                deskewed = self.rotate_image(enhanced, skew_angle)
            else:
                deskewed = enhanced.copy()
        else:
            deskewed = enhanced.copy()

        # 6. Optional Binarization
        final_processed, is_binarized = self.binarize(deskewed, method=options.binarize_method)

        return PreprocessedDocument(
            original=bgr,
            grayscale=gray,
            denoised=denoised,
            contrast_enhanced=enhanced,
            deskewed=deskewed,
            processed=final_processed,
            is_binarized=is_binarized,
            skew_angle=skew_angle,
            original_size=(orig_w, orig_h),
            processed_size=(proc_w, proc_h),
        )

    def normalize_line_image(
        self, line_img: np.ndarray, target_width: Optional[int] = None
    ) -> np.ndarray:
        """Normalize line image strip to fixed height (32px) and normalized float array.

        Args:
            line_img (np.ndarray): Cropped line image.
            target_width (Optional[int]): Fixed width if required.

        Returns:
            np.ndarray: Normalized image array with shape (32, W), values [0.0, 1.0].
        """
        if line_img.ndim == 3:
            line_img = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)

        h, w = line_img.shape[:2]
        if h == 0 or w == 0:
            return np.ones((self.target_line_height, OCR_IMAGE_MIN_WIDTH), dtype=np.float32)

        scale = self.target_line_height / float(h)
        new_w = max(OCR_IMAGE_MIN_WIDTH, int(w * scale))

        if target_width:
            resized = cv2.resize(line_img, (target_width, self.target_line_height), interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(line_img, (new_w, self.target_line_height), interpolation=cv2.INTER_AREA)

        return resized.astype(np.float32) / 255.0
