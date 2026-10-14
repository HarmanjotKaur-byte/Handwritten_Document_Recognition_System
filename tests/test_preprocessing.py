"""Unit tests for ImageProcessor: handwriting-safe preprocessing filters, CLAHE, resizing, and binarization."""

import numpy as np
import pytest
from PIL import Image

from src.preprocessing.image_processor import (
    ImageProcessor,
    PreprocessedDocument,
    PreprocessingOptions,
)


@pytest.fixture
def processor():
    return ImageProcessor(target_line_height=32)


@pytest.fixture
def sample_handwritten_bgr():
    """Create a synthetic handwriting-like canvas with white paper and dark strokes."""
    # 200 x 300 canvas
    canvas = np.ones((200, 300, 3), dtype=np.uint8) * 245
    # Simulate thin pen strokes
    canvas[50:53, 30:250] = (20, 20, 20)
    canvas[100:103, 30:200] = (30, 30, 30)
    canvas[140:170, 80:83] = (25, 25, 25)
    return canvas


# -------------------------------------------------------------
# Tests for Individual Preprocessing Steps
# -------------------------------------------------------------

def test_to_grayscale(processor, sample_handwritten_bgr):
    gray = processor.to_grayscale(sample_handwritten_bgr)
    assert gray.ndim == 2
    assert gray.shape == (200, 300)
    assert gray.dtype == np.uint8

    # Passing already grayscale image should return it intact
    gray_again = processor.to_grayscale(gray)
    assert gray_again.shape == gray.shape


def test_denoise_bilateral_and_gaussian(processor, sample_handwritten_bgr):
    gray = processor.to_grayscale(sample_handwritten_bgr)

    # Bilateral edge-preserving filtering
    bilateral = processor.denoise(gray, method="bilateral")
    assert bilateral.shape == gray.shape
    assert bilateral.dtype == np.uint8
    # Ensure pen stroke locations are still dark
    assert bilateral[51, 100] < 50

    # Gaussian filtering
    gaussian = processor.denoise(gray, method="gaussian")
    assert gaussian.shape == gray.shape
    assert gaussian[51, 100] < 50


def test_enhance_contrast_clahe(processor, sample_handwritten_bgr):
    gray = processor.to_grayscale(sample_handwritten_bgr)
    # Create faint low-contrast band
    gray[120:130, :] = 200

    enhanced = processor.enhance_contrast(gray, clip_limit=2.5)
    assert enhanced.shape == gray.shape
    assert enhanced.dtype == np.uint8
    # CLAHE redistributes local histogram, increasing standard deviation
    assert np.std(enhanced) >= np.std(gray) * 0.9


def test_resize_with_aspect_ratio(processor):
    # Oversized image: 4000 x 2000
    large_img = np.ones((4000, 2000, 3), dtype=np.uint8) * 200
    resized = processor.resize_with_aspect_ratio(large_img, max_dim=1000)

    # Max dimension is height (4000 -> 1000), width should be 500
    assert resized.shape[0] == 1000
    assert resized.shape[1] == 500

    # Image within limit should remain untouched
    small_img = np.ones((500, 600, 3), dtype=np.uint8) * 200
    unchanged = processor.resize_with_aspect_ratio(small_img, max_dim=1000)
    assert unchanged.shape == (500, 600, 3)


def test_binarization_methods(processor, sample_handwritten_bgr):
    gray = processor.to_grayscale(sample_handwritten_bgr)

    # 1. 'none' keeps grayscale values
    out_none, is_bin_none = processor.binarize(gray, method="none")
    assert is_bin_none is False
    assert len(np.unique(out_none)) > 2

    # 2. 'otsu' returns strict binary
    out_otsu, is_bin_otsu = processor.binarize(gray, method="otsu")
    assert is_bin_otsu is True
    assert set(np.unique(out_otsu)).issubset({0, 255})

    # 3. 'adaptive' returns binary
    out_adapt, is_bin_adapt = processor.binarize(gray, method="adaptive")
    assert is_bin_adapt is True
    assert set(np.unique(out_adapt)).issubset({0, 255})

    # 4. 'sauvola' returns binary
    out_sauv, is_bin_sauv = processor.binarize(gray, method="sauvola")
    assert is_bin_sauv is True
    assert set(np.unique(out_sauv)).issubset({0, 255})

    # 5. Invalid method raises ValueError
    with pytest.raises(ValueError, match="Unknown binarization method"):
        processor.binarize(gray, method="invalid_method")


# -------------------------------------------------------------
# Tests for Full Pipeline Execution
# -------------------------------------------------------------

def test_process_page_pipeline_with_options(processor, sample_handwritten_bgr):
    opts = PreprocessingOptions(
        apply_grayscale=True,
        apply_denoise=True,
        denoise_method="bilateral",
        apply_contrast_enhancement=True,
        clahe_clip_limit=2.0,
        apply_deskew=True,
        binarize_method="none",  # enhanced grayscale
        max_dimension=1500,
    )

    doc: PreprocessedDocument = processor.process_page(sample_handwritten_bgr, options=opts)

    assert isinstance(doc, PreprocessedDocument)
    assert doc.original.shape[:2] == (200, 300)
    assert doc.processed.shape == (200, 300)
    assert doc.is_binarized is False
    assert doc.original_size == (300, 200)
    assert doc.processed_size == (300, 200)
    assert isinstance(doc.skew_angle, float)


def test_normalize_line_image(processor):
    crop = np.ones((40, 160), dtype=np.uint8) * 100
    norm = processor.normalize_line_image(crop)
    assert norm.shape[0] == 32
    assert norm.shape[1] == 160 * (32 / 40)
    assert norm.dtype == np.float32
    assert 0.0 <= norm.min() and norm.max() <= 1.0
