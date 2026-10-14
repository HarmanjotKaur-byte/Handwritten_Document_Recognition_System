"""Real PyTorch training and validation pipeline for handwriting recognition on Teklia/IAM-line.

Implements:
- Training and validation split loading from Hugging Face
- Vocabulary extraction from training texts only
- CRNN (CNN + BiLSTM) network training with PyTorch CTCLoss and Adam optimizer
- Tracking of training loss, validation loss, Character Error Rate (CER), and Word Error Rate (WER)
- Automatic checkpoint saving to models/checkpoints/iam_crnn_best.pt based on validation CER
- Smoke test mode for pipeline verification on real samples
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jiwer
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset

from src.recognition.dataset import IAMLineDataset, iam_collate_fn
from src.recognition.decoder import GreedyCTCDecoder
from src.recognition.model import CRNNRecognitionModel
from src.recognition.preprocessing import LineImagePreprocessor
from src.recognition.vocabulary import Vocabulary


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    decoder: GreedyCTCDecoder,
    criterion: nn.CTCLoss,
    device: torch.device,
) -> Tuple[float, float, float, List[Tuple[str, str]]]:
    """Evaluate model on validation split, computing loss, CER, and WER.

    Returns:
        Tuple of (avg_loss, cer, wer, list_of_sample_predictions).
    """
    model.eval()
    total_loss = 0.0
    val_steps = 0

    all_preds: List[str] = []
    all_targets: List[str] = []
    sample_comparisons: List[Tuple[str, str]] = []

    with torch.no_grad():
        for batch_idx, (images, targets, input_lens, target_lens, raw_texts) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device)
            input_lens = input_lens.to(device)
            target_lens = target_lens.to(device)

            # Forward pass: (Batch, Time, num_classes)
            log_probs = model.compute_log_probs(images)

            # Compute CTC loss: nn.CTCLoss expects (Time, Batch, num_classes)
            ctc_log_probs = log_probs.permute(1, 0, 2)
            loss = criterion(ctc_log_probs, targets, input_lens, target_lens)
            if torch.isfinite(loss):
                total_loss += loss.item()
                val_steps += 1

            # Decode predictions
            decoded_batch = decoder.decode(log_probs, input_lengths=input_lens)
            for (pred_text, _), target_text in zip(decoded_batch, raw_texts):
                all_preds.append(pred_text)
                all_targets.append(target_text)
                if len(sample_comparisons) < 5:
                    sample_comparisons.append((pred_text, target_text))

    avg_loss = total_loss / max(1, val_steps)

    # Compute CER and WER using jiwer
    if all_preds and all_targets:
        # Prevent division by zero if all targets are empty
        valid_pairs = [(p, t) for p, t in zip(all_preds, all_targets) if len(t.strip()) > 0]
        if valid_pairs:
            preds_eval, targets_eval = zip(*valid_pairs)
            cer = float(jiwer.cer(list(targets_eval), list(preds_eval)))
            wer = float(jiwer.wer(list(targets_eval), list(preds_eval)))
        else:
            cer, wer = 1.0, 1.0
    else:
        cer, wer = 1.0, 1.0

    return avg_loss, cer, wer, sample_comparisons


def run_smoke_test(
    train_dataset: IAMLineDataset,
    val_dataset: IAMLineDataset,
    vocabulary: Vocabulary,
    device: torch.device,
) -> bool:
    """Smoke test on 20 real IAM samples to verify forward, loss, backward, and optimization."""
    print("\n" + "=" * 60)
    print("RUNNING SMOKE TEST ON REAL IAM-LINE SAMPLES")
    print("=" * 60)

    smoke_train_subset = Subset(train_dataset, list(range(20)))
    smoke_val_subset = Subset(val_dataset, list(range(10)))

    smoke_train_loader = DataLoader(
        smoke_train_subset, batch_size=4, shuffle=True, collate_fn=iam_collate_fn
    )
    smoke_val_loader = DataLoader(
        smoke_val_subset, batch_size=4, shuffle=False, collate_fn=iam_collate_fn
    )

    model = CRNNRecognitionModel(
        num_classes=len(vocabulary),
        in_channels=1,
        lstm_hidden_size=128,
        lstm_num_layers=2,
    ).to(device)

    criterion = nn.CTCLoss(blank=Vocabulary.BLANK_INDEX, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    decoder = GreedyCTCDecoder(vocabulary=vocabulary, blank_index=Vocabulary.BLANK_INDEX)

    print("Executing forward pass, CTC loss calculation, and backward pass...")
    model.train()
    for batch_idx, (images, targets, input_lens, target_lens, raw_texts) in enumerate(smoke_train_loader):
        images = images.to(device)
        targets = targets.to(device)
        input_lens = input_lens.to(device)
        target_lens = target_lens.to(device)

        optimizer.zero_grad()
        log_probs = model.compute_log_probs(images)
        # nn.CTCLoss expects (Time, Batch, num_classes)
        ctc_log_probs = log_probs.permute(1, 0, 2)
        loss = criterion(ctc_log_probs, targets, input_lens, target_lens)

        if not torch.isfinite(loss):
            print(f"[!] Smoke test failed: Non-finite loss encountered: {loss.item()}")
            return False

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        print(f"  Batch {batch_idx + 1}/5 - CTC Loss: {loss.item():.4f}")

    # Test evaluation pass
    val_loss, val_cer, val_wer, samples = evaluate(
        model, smoke_val_loader, decoder, criterion, device
    )
    print(f"Smoke Test Validation - Loss: {val_loss:.4f}, CER: {val_cer:.4f}, WER: {val_wer:.4f}")
    print("Sample Output from Smoke Test:")
    for pred, truth in samples[:2]:
        print(f"  Ground Truth: '{truth}' | Prediction: '{pred}'")

    print("[OK] Smoke test passed successfully! Model architecture, CTC alignment, and optimizer are sound.\n")
    return True


def train_pipeline(args):
    """Main training routine."""
    set_seed(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 60)
    print(f"IAM-Line CRNN Handwriting Recognition Training")
    print(f"Compute Device: {device}")
    print("=" * 60)

    # 1. Load Hugging Face Teklia/IAM-line dataset
    print("\n[1/6] Loading Teklia/IAM-line from Hugging Face...")
    hf_dataset = load_dataset("Teklia/IAM-line")
    train_split = hf_dataset["train"]
    val_split = hf_dataset["validation"]

    print(f"      Train examples: {len(train_split)}")
    print(f"      Validation examples: {len(val_split)}")

    # 2. Build Vocabulary from training split transcriptions ONLY
    print("\n[2/6] Building Character Vocabulary from train split only...")
    train_texts = [str(x["text"] or "") for x in train_split]
    vocabulary = Vocabulary.build_from_texts(train_texts, use_unk=True)
    print(f"      Vocabulary size: {len(vocabulary)} classes (including BLANK and UNK)")

    # Save vocabulary
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = checkpoint_dir / "iam_vocab.json"
    vocabulary.save(vocab_path)
    print(f"      Saved vocabulary to: {vocab_path}")

    # 3. Setup Dataset and Loaders
    print("\n[3/6] Initializing Datasets and DataLoaders...")
    preprocessor = LineImagePreprocessor(
        target_height=32, min_width=32, invert_colors=True
    )

    train_dataset = IAMLineDataset(
        train_split, vocabulary, preprocessor=preprocessor, is_training=True
    )
    val_dataset = IAMLineDataset(
        val_split, vocabulary, preprocessor=preprocessor, is_training=False
    )

    # Optional subset for smoke-testing or constrained training
    if args.max_train_samples and args.max_train_samples < len(train_dataset):
        train_dataset = Subset(train_dataset, list(range(args.max_train_samples)))
        print(f"      Subsetting training data to {args.max_train_samples} samples as requested.")

    if args.max_val_samples and args.max_val_samples < len(val_dataset):
        val_dataset = Subset(val_dataset, list(range(args.max_val_samples)))
        print(f"      Subsetting validation data to {args.max_val_samples} samples as requested.")

    # Run smoke test if requested
    if args.smoke_test:
        smoke_ok = run_smoke_test(train_dataset, val_dataset, vocabulary, device)
        if not smoke_ok:
            sys.exit(1)
        if args.epochs == 0:
            print("Smoke test completed. Exiting as epochs=0.")
            return

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=iam_collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=iam_collate_fn,
        num_workers=0,
    )

    # 4. Model, Loss, Optimizer
    print("\n[4/6] Initializing Model, CTC Loss, and Optimizer...")
    model = CRNNRecognitionModel(
        num_classes=len(vocabulary),
        in_channels=1,
        lstm_hidden_size=args.hidden_size,
        lstm_num_layers=2,
        lstm_dropout=0.2,
    ).to(device)

    criterion = nn.CTCLoss(blank=Vocabulary.BLANK_INDEX, zero_infinity=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    decoder = GreedyCTCDecoder(vocabulary=vocabulary, blank_index=Vocabulary.BLANK_INDEX)

    # Resume from checkpoint if specified
    best_val_cer = float("inf")
    start_epoch = 1
    best_checkpoint_path = checkpoint_dir / args.checkpoint_name

    if args.resume and best_checkpoint_path.is_file():
        print(f"      Resuming from checkpoint: {best_checkpoint_path}")
        ckpt = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_cer = ckpt.get("val_cer", float("inf"))
        print(f"      Resumed epoch {start_epoch}, previous best CER: {best_val_cer:.4f}")

    # 5. Training Loop
    print(f"\n[5/6] Commencing Training for {args.epochs} Epochs...")
    total_start_time = time.time()

    for epoch in range(start_epoch, start_epoch + args.epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        train_steps = 0

        for batch_idx, (images, targets, input_lens, target_lens, raw_texts) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            input_lens = input_lens.to(device)
            target_lens = target_lens.to(device)

            optimizer.zero_grad()
            log_probs = model.compute_log_probs(images)
            # nn.CTCLoss expects (Time, Batch, num_classes)
            ctc_log_probs = log_probs.permute(1, 0, 2)
            loss = criterion(ctc_log_probs, targets, input_lens, target_lens)

            if not torch.isfinite(loss):
                print(f"[Warning] Non-finite loss at epoch {epoch}, batch {batch_idx}. Skipping.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                print(
                    f"  Epoch [{epoch}/{start_epoch + args.epochs - 1}] | "
                    f"Batch [{batch_idx + 1}/{len(train_loader)}] | "
                    f"Train Loss: {train_loss / max(1, train_steps):.4f}"
                )

        avg_train_loss = train_loss / max(1, train_steps)

        # Validation
        val_loss, val_cer, val_wer, sample_pairs = evaluate(
            model, val_loader, decoder, criterion, device
        )
        scheduler.step(val_cer)

        epoch_duration = time.time() - epoch_start
        print(f"\n>> Epoch {epoch} Summary ({epoch_duration:.1f}s):")
        print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val CER: {val_cer:.4f} ({val_cer:.1%}) | Val WER: {val_wer:.4f} ({val_wer:.1%})")

        print("   Sample Validation Predictions:")
        for pred, truth in sample_pairs[:3]:
            print(f"     [Truth]: '{truth}'\n     [Pred ]: '{pred}'")

        # Save Best Checkpoint based on Validation CER
        if val_cer < best_val_cer:
            best_val_cer = val_cer
            checkpoint_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config": model.get_config(),
                "vocabulary": vocabulary.to_dict(),
                "preprocessing_config": preprocessor.get_config(),
                "val_loss": val_loss,
                "val_cer": val_cer,
                "val_wer": val_wer,
            }
            torch.save(checkpoint_payload, best_checkpoint_path)
            print(f"   [*] Saved new BEST checkpoint (CER: {val_cer:.4f}) to: {best_checkpoint_path}")

        print("-" * 60)

    total_duration = time.time() - total_start_time
    print(f"\n[6/6] Training completed in {total_duration / 60:.2f} minutes.")
    print(f"      Best Validation CER: {best_val_cer:.4f} ({best_val_cer:.1%})")
    print(f"      Checkpoint saved at: {best_checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IAM-Line CRNN Handwriting Model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=256, help="BiLSTM hidden size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="", help="Device: 'cuda' or 'cpu'")
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints", help="Directory for checkpoints")
    parser.add_argument("--checkpoint-name", type=str, default="iam_crnn_best.pt", help="Checkpoint filename")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit train samples for fast training")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Limit val samples for fast validation")
    parser.add_argument("--smoke-test", action="store_true", help="Run initial smoke test on 20 samples")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if found")

    args = parser.parse_args()
    train_pipeline(args)
