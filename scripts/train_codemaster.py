"""Training and fine-tuning script for CodeMaster-CRNN on IAM Words Dataset.

Supports:
- Pretrained weight initialization (from CodeMaster-AJ best_model.pth)
- Training up to arbitrary epochs (e.g., 100 or 500) with early stopping
- Learning rate scheduling and gradient clipping
- Validation loss and Character Error Rate (CER) calculation per epoch
- Loss history recording and checkpoint saving
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recognition.model import CodeMasterCRNN

# Standard 76-character alphabet used in CodeMaster-AJ
DEFAULT_ALPHABET = "!\"#'()*,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def parse_iam_dataset(txt_path: str, img_dir: str):
    """Parse IAM words dataset annotations."""
    samples = []
    chars = set()

    with open(txt_path, "r", encoding="latin1") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 9 or parts[1] != "ok":
                continue
            file_id = parts[0]
            word_text = parts[-1]
            id_parts = file_id.split("-")
            sub1 = id_parts[0]
            sub2 = f"{id_parts[0]}-{id_parts[1]}"
            nested_img_path = os.path.join(img_dir, sub1, sub2, f"{file_id}.png")

            if os.path.exists(nested_img_path):
                samples.append((nested_img_path, word_text))
                for char in word_text:
                    chars.add(char)

    alphabet = sorted(list(chars))
    return samples, alphabet


class IAMWordDataset(Dataset):
    """PyTorch Dataset for IAM word images with CodeMaster standard preprocessing."""

    def __init__(self, samples, char_to_num, img_w=128, img_h=32):
        self.samples = samples
        self.char_to_num = char_to_num
        self.img_w = img_w
        self.img_h = img_h

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, word = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.ones((self.img_h, self.img_w), dtype=np.uint8) * 255

        h, w = img.shape
        fx = self.img_w / max(1, w)
        fy = self.img_h / max(1, h)
        f = min(fx, fy)
        new_w, new_h = max(1, int(w * f)), max(1, int(h * f))
        img_resized = cv2.resize(img, (new_w, new_h))

        canvas = np.ones((self.img_h, self.img_w), dtype=np.uint8) * 255
        canvas[0:new_h, 0:new_w] = img_resized

        img_normalized = canvas.astype(np.float32) / 255.0
        img_tensor = np.expand_dims(img_normalized, axis=0)  # Shape: (1, 32, 128)

        # Map text to integer labels (skipping unmapped characters)
        label_list = [self.char_to_num[char] for char in word if char in self.char_to_num]
        if not label_list:
            label_list = [self.char_to_num.get("a", 1)]

        return torch.tensor(img_tensor, dtype=torch.float32), label_list


def ctc_collate(batch):
    """Collate function flattening targets for PyTorch nn.CTCLoss."""
    images = []
    targets = []
    target_lengths = []
    for img, labels in batch:
        images.append(img)
        targets.extend(labels)
        target_lengths.append(len(labels))

    return (
        torch.stack(images, dim=0),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(target_lengths, dtype=torch.long),
    )


def decode_predictions(pred_indexes, num_to_char):
    """Greedy CTC decode sequence of indexes."""
    decoded_string = ""
    last_seen_token = None
    for token in pred_indexes:
        if token != 0 and token != last_seen_token:
            decoded_string += num_to_char.get(token, "")
        last_seen_token = token
    return decoded_string


def compute_cer(ref: str, hyp: str) -> float:
    """Levenshtein distance CER."""
    r, h = list(ref), list(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    for i in range(len(r) + 1):
        d[i, 0] = i
    for j in range(len(h) + 1):
        d[0, j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = 1 + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return float(d[len(r), len(h)]) / max(1, len(r))


def main():
    parser = argparse.ArgumentParser(description="Train / Fine-tune CodeMaster-CRNN")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit training samples for testing")
    parser.add_argument("--max_val_samples", type=int, default=500, help="Limit validation samples")
    parser.add_argument("--pretrained", type=str, default="models/checkpoints/codemaster_crnn_best.pth", help="Pretrained weights path")
    parser.add_argument("--output_path", type=str, default="models/checkpoints/codemaster_crnn_finetuned.pth", help="Output model path")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    args = parser.parse_args()

    txt_path = "external/handwritten-text-recognition/iam_words/words.txt"
    img_dir = "external/handwritten-text-recognition/iam_words/words"

    if not os.path.exists(txt_path) or not os.path.exists(img_dir):
        print(f"Error: IAM words dataset not found at {txt_path} or {img_dir}")
        sys.exit(1)

    print(f"Parsing IAM dataset annotations from {txt_path}...")
    samples, alphabet_found = parse_iam_dataset(txt_path, img_dir)
    print(f"Found {len(samples)} valid word images.")

    # Use default 76-character alphabet to ensure checkpoint compatibility
    alphabet = list(DEFAULT_ALPHABET)
    char_to_num = {char: idx + 1 for idx, char in enumerate(alphabet)}
    num_to_char = {idx + 1: char for idx, char in enumerate(alphabet)}
    num_classes = len(alphabet) + 1  # 77 classes

    split_idx = int(len(samples) * 0.95)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    if args.max_train_samples:
        train_samples = train_samples[: args.max_train_samples]
    if args.max_val_samples:
        val_samples = val_samples[: args.max_val_samples]

    print(f"Dataset splits: {len(train_samples)} training samples, {len(val_samples)} validation samples.")

    train_dataset = IAMWordDataset(train_samples, char_to_num)
    val_dataset = IAMWordDataset(val_samples, char_to_num)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=ctc_collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ctc_collate,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CodeMasterCRNN(num_classes=num_classes).to(device)

    # Load pretrained weights if provided
    if args.pretrained and os.path.isfile(args.pretrained):
        print(f"Loading pretrained weights from {args.pretrained}...")
        try:
            ckpt = torch.load(args.pretrained, map_location=device)
            # Support both direct state_dict and dict with 'model_state_dict'
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                ckpt = ckpt["model_state_dict"]
            model.load_state_dict(ckpt)
            print("Pretrained weights loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights ({e}). Initializing from scratch.")

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_cer": [], "val_acc": []}

    print("\nStarting Training / Fine-Tuning Loop...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for images, targets, target_lengths in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # (Batch, 32, num_classes)
            outputs_ctc = outputs.permute(1, 0, 2)  # (32, Batch, num_classes)

            batch_size = images.size(0)
            input_lengths = torch.full(
                size=(batch_size,), fill_value=32, dtype=torch.long, device=device
            )

            loss = criterion(outputs_ctc, targets, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item() * batch_size

        epoch_train_loss = train_loss / max(1, len(train_dataset))

        # Validation Phase
        model.eval()
        val_loss = 0.0
        total_cer = 0.0
        exact_matches = 0
        total_val_words = 0

        with torch.no_grad():
            for images, targets, target_lengths in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)

                outputs = model(images)
                outputs_ctc = outputs.permute(1, 0, 2)
                batch_size = images.size(0)
                input_lengths = torch.full(
                    size=(batch_size,), fill_value=32, dtype=torch.long, device=device
                )

                loss = criterion(outputs_ctc, targets, input_lengths, target_lengths)
                val_loss += loss.item() * batch_size

                # Greedy decode and metric calculation
                pred_indexes = torch.argmax(outputs, dim=2).cpu().numpy()
                target_arr = targets.cpu().numpy()
                target_lens = target_lengths.cpu().numpy()

                offset = 0
                for b in range(batch_size):
                    l = target_lens[b]
                    ref_chars = "".join(num_to_char.get(c, "") for c in target_arr[offset : offset + l])
                    offset += l

                    hyp_chars = decode_predictions(pred_indexes[b], num_to_char)
                    cer = compute_cer(ref_chars, hyp_chars)
                    total_cer += cer
                    if hyp_chars.lower() == ref_chars.lower():
                        exact_matches += 1
                    total_val_words += 1

        epoch_val_loss = val_loss / max(1, len(val_dataset))
        epoch_cer = total_cer / max(1, total_val_words)
        epoch_acc = exact_matches / max(1, total_val_words)
        elapsed = time.time() - t0

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_cer"].append(epoch_cer)
        history["val_acc"].append(epoch_acc)

        print(
            f"Epoch {epoch:3d}/{args.epochs:3d} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val CER: {epoch_cer:.2%} | "
            f"Val Exact Acc: {epoch_acc:.2%} | "
            f"Time: {elapsed:.1f}s"
        )

        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            torch.save(model.state_dict(), args.output_path)
            print(f"  --> Saved new best checkpoint to {args.output_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # Save training history JSON
    history_path = os.path.splitext(args.output_path)[0] + "_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training completed. Metrics saved to {history_path}")


if __name__ == "__main__":
    main()
