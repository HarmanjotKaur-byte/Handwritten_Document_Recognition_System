"""Document loading and validation module supporting image files and multi-page PDFs.

Validates file extensions, file sizes, and payload integrity. Converts multi-page
PDF documents to high-resolution images using PyMuPDF (fitz) and decodes PNG/JPG/JPEG images.
"""

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, List, Set, Union

from PIL import Image

# Accepted document and image extensions
ALLOWED_IMAGE_EXTENSIONS: Set[str] = {".png", ".jpg", ".jpeg"}
ALLOWED_PDF_EXTENSIONS: Set[str] = {".pdf"}
ALLOWED_EXTENSIONS: Set[str] = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_PDF_EXTENSIONS

# Default max file size limit: 25 Megabytes
DEFAULT_MAX_FILE_SIZE_BYTES: int = 25 * 1024 * 1024


class DocumentValidationError(ValueError):
    """Custom exception raised when an ingested document fails validation checks."""
    pass


@dataclass
class LoadedDocument:
    """Container holding ingested document metadata and rasterized PIL page images."""

    filename: str
    is_pdf: bool
    page_count: int
    pages: List[Image.Image]  # PIL images in RGB mode for each page
    file_size_bytes: int = 0


class DocumentLoader:
    """Validates and loads document images and PDFs into standardized in-memory PIL Images."""

    def __init__(
        self,
        target_dpi: int = 150,
        max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    ):
        """Initialize DocumentLoader.

        Args:
            target_dpi (int): Rendering resolution (DPI) for rasterizing PDF pages.
            max_file_size_bytes (int): Maximum allowable upload file size in bytes.
        """
        self.target_dpi = target_dpi
        self.max_file_size_bytes = max_file_size_bytes

    def validate_file(self, filename: str, data_len: int) -> None:
        """Perform validation on filename suffix and content length.

        Args:
            filename (str): Name of the file with extension.
            data_len (int): Length of byte buffer.

        Raises:
            DocumentValidationError: If file is empty, exceeds size limit, or has unsupported extension.
        """
        if data_len == 0:
            raise DocumentValidationError("Upload error: The provided file is empty (0 bytes).")

        if data_len > self.max_file_size_bytes:
            max_mb = self.max_file_size_bytes / (1024 * 1024)
            current_mb = data_len / (1024 * 1024)
            raise DocumentValidationError(
                f"File size limit exceeded: Upload is {current_mb:.2f} MB, which exceeds maximum limit of {max_mb:.1f} MB."
            )

        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            allowed_list = ", ".join(sorted(ALLOWED_EXTENSIONS))
            raise DocumentValidationError(
                f"Unsupported file format '{suffix}'. Allowed formats: {allowed_list}"
            )

    def load_from_path(self, file_path: Union[str, Path]) -> LoadedDocument:
        """Load and validate document from a local filesystem path.

        Args:
            file_path (Union[str, Path]): Local file path.

        Returns:
            LoadedDocument: Ingested document with extracted pages.

        Raises:
            FileNotFoundError: If path does not exist.
            DocumentValidationError: If file fails validation checks.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found on filesystem: {path}")

        file_size = path.stat().st_size
        self.validate_file(path.name, file_size)

        with open(path, "rb") as f:
            return self.load_from_bytes(f, filename=path.name)

    def load_from_bytes(
        self, file_stream: Union[BinaryIO, bytes], filename: str = "document"
    ) -> LoadedDocument:
        """Validate and load document from a byte buffer or stream.

        Args:
            file_stream (Union[BinaryIO, bytes]): Raw byte data or stream.
            filename (str): Original file name for format identification.

        Returns:
            LoadedDocument: Container with rasterized/decoded PIL Images.

        Raises:
            DocumentValidationError: If validation fails or decoding fails.
        """
        if isinstance(file_stream, bytes):
            data = file_stream
        else:
            data = file_stream.read()

        data_len = len(data) if data else 0
        self.validate_file(filename, data_len)

        suffix = Path(filename).suffix.lower()

        if suffix in ALLOWED_PDF_EXTENSIONS:
            pages = self._render_pdf_bytes(data, filename)
            return LoadedDocument(
                filename=filename,
                is_pdf=True,
                page_count=len(pages),
                pages=pages,
                file_size_bytes=data_len,
            )
        elif suffix in ALLOWED_IMAGE_EXTENSIONS:
            # Decode image via Pillow
            try:
                img = Image.open(io.BytesIO(data))
                img.load()  # Force decode to detect corrupted payloads
                # Ensure 3-channel RGB format
                if img.mode != "RGB":
                    img = img.convert("RGB")
                return LoadedDocument(
                    filename=filename,
                    is_pdf=False,
                    page_count=1,
                    pages=[img],
                    file_size_bytes=data_len,
                )
            except Exception as e:
                raise DocumentValidationError(
                    f"Corrupted or invalid image data in '{filename}': {e}"
                ) from e
        else:
            raise DocumentValidationError(f"Unsupported file format: {suffix}")

    def _render_pdf_bytes(self, pdf_bytes: bytes, filename: str = "document.pdf") -> List[Image.Image]:
        """Rasterize PDF pages into high-resolution RGB PIL Images using PyMuPDF.

        Args:
            pdf_bytes (bytes): Binary PDF data.
            filename (str): Name of the file for reporting.

        Returns:
            List[Image.Image]: List of rendered PIL Images.

        Raises:
            DocumentValidationError: If PDF is corrupted or has 0 pages.
        """
        import fitz  # PyMuPDF

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            raise DocumentValidationError(
                f"Failed to open PDF document '{filename}'. File may be corrupted or encrypted: {e}"
            ) from e

        # Calculate zoom matrix based on target DPI (standard PDF point is 72 dpi)
        zoom = self.target_dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pages: List[Image.Image] = []

        try:
            total_pages = len(doc)
            if total_pages == 0:
                raise DocumentValidationError(f"PDF document '{filename}' contains 0 renderable pages.")

            for page_index in range(total_pages):
                page = doc.load_page(page_index)
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
                pages.append(img)
        except Exception as e:
            if not isinstance(e, DocumentValidationError):
                raise DocumentValidationError(f"Error rendering pages of PDF '{filename}': {e}") from e
            raise
        finally:
            doc.close()

        return pages
