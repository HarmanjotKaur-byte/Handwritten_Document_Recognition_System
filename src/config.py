"""Configuration and environment management for the application.

Loads environment variables, defines filesystem paths, system defaults,
and constants used across all modules.
"""

import os
from pathlib import Path
from typing import Set

# Root project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data and artifact storage directories
DATA_DIR = BASE_DIR / "data"
DATABASE_DIR = DATA_DIR / "database"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_documents"
MODELS_DIR = BASE_DIR / "models"

# Ensure runtime directories exist
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# SQLite Database path
DEFAULT_DB_PATH = DATABASE_DIR / "documents.db"
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(DEFAULT_DB_PATH)))

# Model file paths
OCR_MODEL_PATH = Path(os.getenv("OCR_MODEL_PATH", str(MODELS_DIR / "crnn_handwriting.pth")))
CLASSIFIER_MODEL_PATH = Path(
    os.getenv("CLASSIFIER_MODEL_PATH", str(MODELS_DIR / "document_classifier.joblib"))
)
VECTORIZER_MODEL_PATH = Path(
    os.getenv("VECTORIZER_MODEL_PATH", str(MODELS_DIR / "tfidf_vectorizer.joblib"))
)

# Supported ingestion file extensions
SUPPORTED_IMAGE_EXTENSIONS: Set[str] = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
SUPPORTED_DOCUMENT_EXTENSIONS: Set[str] = {".pdf"}
SUPPORTED_EXTENSIONS: Set[str] = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_DOCUMENT_EXTENSIONS

# CRNN Architecture & Preprocessing parameters
OCR_IMAGE_HEIGHT: int = int(os.getenv("OCR_IMAGE_HEIGHT", "32"))
OCR_IMAGE_MIN_WIDTH: int = 32
OCR_BATCH_SIZE: int = int(os.getenv("OCR_BATCH_SIZE", "8"))

# Character set for CTC recognition: Alphanumeric + standard punctuation
# Index 0 is reserved for the CTC Blank token in PyTorch
OCR_CHARACTERS: str = (
    "0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
)

# Classification categories
DOCUMENT_CLASSES = [
    "Medical Record",
    "Invoice / Receipt",
    "Application Form",
    "Handwritten Note / Letter",
    "General / Unclassified",
]
CLASSIFIER_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("CLASSIFIER_CONFIDENCE_THRESHOLD", "0.35")
)

# spaCy language model name
SPACY_MODEL_NAME: str = os.getenv("SPACY_MODEL_NAME", "en_core_web_sm")


def get_torch_device() -> str:
    """Determine the optimal PyTorch device (CUDA or CPU).

    Returns:
        str: 'cuda' if CUDA is available and requested, else 'cpu'.
    """
    requested_device = os.getenv("DEVICE", "auto").lower()
    if requested_device == "cuda":
        return "cuda"
    elif requested_device == "cpu":
        return "cpu"

    # Auto-detection
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
