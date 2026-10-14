"""General helper functions for datatype formatting and conversions."""

from datetime import datetime
from typing import Union

import cv2
import numpy as np
from PIL import Image


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert a NumPy image array (BGR, RGB, or Grayscale) to a PIL Image.

    Args:
        arr (np.ndarray): NumPy image array.

    Returns:
        Image.Image: PIL RGB or L image.
    """
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    elif arr.ndim == 3:
        if arr.shape[2] == 3:
            # Assume OpenCV BGR by default
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb, mode="RGB")
        elif arr.shape[2] == 4:
            rgba = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
            return Image.fromarray(rgba, mode="RGBA")
    raise ValueError(f"Invalid image shape for conversion: {arr.shape}")


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert a PIL Image to a NumPy BGR image array.

    Args:
        img (Image.Image): PIL Image.

    Returns:
        np.ndarray: OpenCV BGR image array.
    """
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def format_file_size(num_bytes: int) -> str:
    """Format byte count into human-readable unit string.

    Args:
        num_bytes (int): Size in bytes.

    Returns:
        str: Formatted string (e.g., '1.24 MB').
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def format_timestamp(iso_str: str) -> str:
    """Format an ISO 8601 timestamp string into a readable datetime.

    Args:
        iso_str (str): ISO formatted datetime.

    Returns:
        str: Readable date and time string.
    """
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%b %d, %Y - %I:%M %p")
    except Exception:
        return iso_str
