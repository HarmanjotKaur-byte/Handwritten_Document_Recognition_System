"""Unit tests for DocumentLoader: validating file types, sizes, empty uploads, and PDF conversion."""

import io
import fitz  # PyMuPDF
import pytest
from PIL import Image

from src.io.loader import (
    ALLOWED_EXTENSIONS,
    DocumentLoader,
    DocumentValidationError,
    LoadedDocument,
)


@pytest.fixture
def loader():
    """Provides DocumentLoader instance with 5MB limit for testing."""
    return DocumentLoader(target_dpi=150, max_file_size_bytes=5 * 1024 * 1024)


def create_dummy_image_bytes(format_name: str = "PNG", size=(120, 80)) -> bytes:
    """Helper to generate in-memory dummy image bytes."""
    img = Image.new("RGB", size, color=(240, 240, 240))
    buf = io.BytesIO()
    img.save(buf, format=format_name)
    return buf.getvalue()


def create_dummy_pdf_bytes(num_pages: int = 2) -> bytes:
    """Helper to generate in-memory multi-page PDF bytes using PyMuPDF."""
    doc = fitz.open()
    for p in range(num_pages):
        page = doc.new_page(width=300, height=400)
        page.insert_text((50, 100), f"Handwritten Sample Page {p + 1}", fontsize=14)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# -------------------------------------------------------------
# Tests for Valid Inputs
# -------------------------------------------------------------

def test_load_valid_png_image(loader):
    png_bytes = create_dummy_image_bytes("PNG")
    doc = loader.load_from_bytes(png_bytes, filename="sample.png")
    assert isinstance(doc, LoadedDocument)
    assert not doc.is_pdf
    assert doc.page_count == 1
    assert len(doc.pages) == 1
    assert doc.pages[0].size == (120, 80)
    assert doc.file_size_bytes == len(png_bytes)


def test_load_valid_jpg_and_jpeg_images(loader):
    jpg_bytes = create_dummy_image_bytes("JPEG")
    doc_jpg = loader.load_from_bytes(jpg_bytes, filename="sample.jpg")
    assert doc_jpg.page_count == 1
    assert not doc_jpg.is_pdf

    doc_jpeg = loader.load_from_bytes(jpg_bytes, filename="sample.jpeg")
    assert doc_jpeg.page_count == 1
    assert not doc_jpeg.is_pdf


def test_load_valid_multi_page_pdf(loader):
    pdf_bytes = create_dummy_pdf_bytes(num_pages=3)
    doc = loader.load_from_bytes(pdf_bytes, filename="multipage_document.pdf")
    assert isinstance(doc, LoadedDocument)
    assert doc.is_pdf is True
    assert doc.page_count == 3
    assert len(doc.pages) == 3
    # Check rendered PIL Image integrity
    for page in doc.pages:
        assert isinstance(page, Image.Image)
        assert page.mode == "RGB"
        assert page.width > 0 and page.height > 0


def test_load_from_filesystem_path(loader, tmp_path):
    img_bytes = create_dummy_image_bytes("PNG")
    file_path = tmp_path / "saved_doc.png"
    file_path.write_bytes(img_bytes)

    loaded = loader.load_from_path(file_path)
    assert loaded.filename == "saved_doc.png"
    assert loaded.page_count == 1


# -------------------------------------------------------------
# Tests for Invalid Inputs and Edge Cases
# -------------------------------------------------------------

def test_empty_upload_raises_validation_error(loader):
    with pytest.raises(DocumentValidationError, match="empty"):
        loader.load_from_bytes(b"", filename="empty.png")


def test_unsupported_file_extension_raises_validation_error(loader):
    with pytest.raises(DocumentValidationError, match="Unsupported file format"):
        loader.load_from_bytes(b"some content", filename="report.txt")

    with pytest.raises(DocumentValidationError, match="Unsupported file format"):
        loader.load_from_bytes(b"some content", filename="doc.docx")


def test_file_size_limit_exceeded(loader):
    # Set limit to 100 KB
    strict_loader = DocumentLoader(max_file_size_bytes=100 * 1024)
    oversized_data = b"x" * (120 * 1024)

    with pytest.raises(DocumentValidationError, match="File size limit exceeded"):
        strict_loader.load_from_bytes(oversized_data, filename="large.png")


def test_corrupted_image_data_raises_validation_error(loader):
    corrupt_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00corrupted_data_not_an_image"
    with pytest.raises(DocumentValidationError, match="Corrupted or invalid image"):
        loader.load_from_bytes(corrupt_bytes, filename="corrupt.png")


def test_corrupted_pdf_data_raises_validation_error(loader):
    corrupt_pdf = b"%PDF-1.4\ncorrupted random bytes without trailer"
    with pytest.raises(DocumentValidationError, match="Failed to open PDF document"):
        loader.load_from_bytes(corrupt_pdf, filename="bad.pdf")


def test_load_from_nonexistent_path_raises_file_not_found(loader, tmp_path):
    missing_file = tmp_path / "does_not_exist.png"
    with pytest.raises(FileNotFoundError):
        loader.load_from_path(missing_file)
