"""Training and fine-tuning pipeline for the PyTorch CRNN handwriting recognition model.

Supports training on line crops using CTCLoss with the Adam optimizer.
Can generate synthetic line samples for initial sanity verification and benchmark training.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure project root is in sys.path when invoked directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import MODELS_DIR, OCR_CHARACTERS, OCR_MODEL_PATH, get_torch_device
from src.ocr.dataset import LineCropDataset, pad_collate_fn
from src.ocr.decoder import CTCDecoder
from src.ocr.model import CRNN
from src.preprocessing.image_processor import ImageProcessor


def generate_synthetic_line(
    text: str, height: int = 32, font_scale: float = 0.7
) -> np.ndarray:
    """Generate a synthetic text line crop for pipeline verification and baseline training.

    Args:
        text (str): Ground-truth string to render.
        height (int): Line image height.
        font_scale (float): OpenCV font scale.

    Returns:
        np.ndarray: Grayscale line image.
    """
    # Estimate width
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, 2)
    img_w = max(64, text_w + 30)

    # White background with black text
    canvas = np.ones((height, img_w), dtype=np.uint8) * 255
    y_pos = int((height + text_h) / 2)
    cv2.putText(canvas, text, (15, y_pos), font, font_scale, (0,), 2, cv2.LINE_AA)

    # Add slight random noise/blur
    noise = np.random.normal(0, 5, canvas.shape).astype(np.int16)
    noisy_canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return noisy_canvas


def create_synthetic_dataset(count: int = 100) -> List[Tuple[np.ndarray, str]]:
    """Build a list of synthetic line image and text pairs."""
    vocabulary_phrases = [
        "Patient John Doe",
        "Prescription Date 2024-10-15",
        "Total Amount USD 450.00",
        "Invoice Number INV-8921",
        "Phone number 555-0199",
        "Email support@company.org",
        "Doctor Jane Smith",
        "Diagnosis Hypertension",
        "Payment received with thanks",
        "Application Form Verified",
    ]
    samples: List[Tuple[np.ndarray, str]] = []
    for i in range(count):
        phrase = vocabulary_phrases[i % len(vocabulary_phrases)]
        img = generate_synthetic_line(phrase)
        samples.append((img, phrase))
    return samples


def train(
    epochs: int = 15,
    batch_size: int = 8,
    learning_rate: float = 0.0005,
    output_path: Path = OCR_MODEL_PATH,
) -> None:
    """Train or fine-tune CRNN model with CTC loss.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Mini-batch size.
        learning_rate (float): Optimizer learning rate.
        output_path (Path): Path to save trained state_dict.
    """
    device = torch.device(get_torch_device())
    print(f"[CRNN Training] Using compute device: {device}")

    decoder = CTCDecoder(OCR_CHARACTERS)
    processor = ImageProcessor()
    model = CRNN(num_classes=decoder.num_classes).to(device)

    # Initialize PyTorch CTC Loss
    # zero_infinity=True prevents crashes if targets are longer than inputs
    ctc_loss = nn.CTCLoss(blank=decoder.blank_idx, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("[CRNN Training] Generating benchmark training samples...")
    train_samples = create_synthetic_dataset(count=120)
    dataset = LineCropDataset(train_samples, decoder, processor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)

    print(f"[CRNN Training] Commencing training for {epochs} epochs...")
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_idx, (imgs, targets, feature_lens, target_lens) in enumerate(loader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            # Model forward: (Time, Batch, num_classes)
            logits = model(imgs)
            log_probs = logits.log_softmax(2)

            # PyTorch CTCLoss expects (input_lengths, target_lengths)
            loss = ctc_loss(log_probs, targets, feature_lens, target_lens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{epochs}] - CTC Loss: {avg_loss:.4f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"[CRNN Training] Trained weights saved successfully to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CRNN Handwriting Model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
