"""Evaluation script for the trained IAM CRNN handwriting model on the official test split."""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jiwer
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.recognition.dataset import IAMLineDataset, iam_collate_fn
from src.recognition.decoder import GreedyCTCDecoder
from src.recognition.model import CRNNRecognitionModel
from src.recognition.preprocessing import LineImagePreprocessor
from src.recognition.vocabulary import Vocabulary


def evaluate_checkpoint(
    checkpoint_path: Path,
    max_test_samples: int = None,
    batch_size: int = 16,
    device_name: str = "",
):
    """Evaluate trained checkpoint on Teklia/IAM-line test split."""
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))

    print("=" * 60)
    print("IAM-Line Handwriting Model Evaluation on Official Test Split")
    print(f"Compute Device: {device}")
    print(f"Checkpoint Path: {checkpoint_path}")
    print("=" * 60)

    if not checkpoint_path.is_file():
        print(f"[!] Error: Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)

    print("\n[1/4] Loading checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Reconstruct vocabulary
    vocab_dict = ckpt["vocabulary"]
    vocabulary = Vocabulary.from_dict(vocab_dict)
    print(f"      Reconstructed Vocabulary ({len(vocabulary)} classes, blank index={Vocabulary.BLANK_INDEX})")

    # Reconstruct preprocessor
    prep_cfg = ckpt.get("preprocessing_config", {})
    preprocessor = LineImagePreprocessor(
        target_height=prep_cfg.get("target_height", 32),
        min_width=prep_cfg.get("min_width", 32),
        invert_colors=prep_cfg.get("invert_colors", True),
    )

    # Reconstruct model architecture
    model_cfg = ckpt.get("model_config", {})
    model = CRNNRecognitionModel(
        num_classes=len(vocabulary),
        in_channels=model_cfg.get("in_channels", 1),
        lstm_hidden_size=model_cfg.get("lstm_hidden_size", 256),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("      Model loaded in evaluation mode.")

    # 2. Load test split
    print("\n[2/4] Loading Teklia/IAM-line official TEST split...")
    hf_dataset = load_dataset("Teklia/IAM-line")
    test_split = hf_dataset["test"]
    print(f"      Total official test samples: {len(test_split)}")

    test_dataset = IAMLineDataset(
        test_split, vocabulary, preprocessor=preprocessor, is_training=False
    )

    if max_test_samples and max_test_samples < len(test_dataset):
        test_dataset = Subset(test_dataset, list(range(max_test_samples)))
        print(f"      Evaluating subset of {max_test_samples} test samples.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=iam_collate_fn,
        num_workers=0,
    )

    decoder = GreedyCTCDecoder(vocabulary=vocabulary, blank_index=Vocabulary.BLANK_INDEX)

    # 3. Inference & metric collection
    print("\n[3/4] Running inference over test samples...")
    all_preds: List[str] = []
    all_targets: List[str] = []
    comparisons: List[Tuple[str, str]] = []
    skipped_count = 0

    with torch.no_grad():
        for images, targets, input_lens, target_lens, raw_texts in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            input_lens = input_lens.to(device)

            log_probs = model.compute_log_probs(images)
            decoded = decoder.decode(log_probs, input_lengths=input_lens)

            for (pred_text, _), truth_text in zip(decoded, raw_texts):
                if not truth_text.strip():
                    skipped_count += 1
                    continue
                all_preds.append(pred_text)
                all_targets.append(truth_text)
                if len(comparisons) < 10:
                    comparisons.append((pred_text, truth_text))

    # 4. Compute Metrics
    print("\n[4/4] Computing Official Metrics (CER & WER)...")
    if all_preds and all_targets:
        cer = float(jiwer.cer(all_targets, all_preds))
        wer = float(jiwer.wer(all_targets, all_preds))
    else:
        cer, wer = 1.0, 1.0

    print("=" * 60)
    print("EVALUATION RESULTS REPORT:")
    print(f"  Total Test Samples Evaluated: {len(all_preds)}")
    print(f"  Skipped Samples (empty labels): {skipped_count}")
    print(f"  Character Error Rate (CER): {cer:.4f} ({cer:.2%})")
    print(f"  Word Error Rate (WER):      {wer:.4f} ({wer:.2%})")
    print("=" * 60)

    print("\nSample Test Predictions vs. Ground Truth:")
    for i, (pred, truth) in enumerate(comparisons, 1):
        print(f"\n  Sample #{i}:")
        print(f"    [Ground Truth]: '{truth}'")
        print(f"    [Predicted   ]: '{pred}'")

    return cer, wer, len(all_preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate IAM-Line CRNN Model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/iam_crnn_best.pt",
        help="Path to trained checkpoint",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of test samples")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="", help="Device: 'cuda' or 'cpu'")

    args = parser.parse_args()
    evaluate_checkpoint(
        Path(args.checkpoint),
        max_test_samples=args.max_samples,
        batch_size=args.batch_size,
        device_name=args.device,
    )
