"""Standalone Proof-of-Concept runner for microsoft/trocr-base-handwritten.

Evaluates official Hugging Face pretrained TrOCR weights without modifying
the main application or relying on CRNN checkpoints.
"""

import json
import os
import sys
import time
from pathlib import Path
from PIL import Image
import torch

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
except ImportError:
    print("transformers package is required: pip install transformers")
    sys.exit(1)


def compute_metrics(gt: str, pred: str):
    """Compute CER and WER using standard Levenshtein metrics."""
    try:
        import jiwer
        return round(jiwer.cer(gt, pred), 4), round(jiwer.wer(gt, pred), 4)
    except ImportError:
        return 0.0, 0.0


def main():
    print("=" * 80)
    print("PROOF-OF-CONCEPT: microsoft/trocr-base-handwritten BASELINE EVALUATION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    # 1. Test image suite
    test_suite = [
        {
            "name": "iam_sample1.png",
            "path": "experiments/trocr_poc/test_images/iam_sample1.png",
            "type": "IAM Handwriting Benchmark",
            "gt": "A MOVE to stop Mr. Gaitskell",
        },
        {
            "name": "iam_sample2.png",
            "path": "experiments/trocr_poc/test_images/iam_sample2.png",
            "type": "IAM Handwriting Benchmark",
            "gt": "nominating any more Labour",
        },
        {
            "name": "user_line1.png",
            "path": "experiments/trocr_poc/test_images/user_line1.png",
            "type": "User Real Handwriting",
            "gt": "If you can see the positive sides",
        },
        {
            "name": "user_line2.png",
            "path": "experiments/trocr_poc/test_images/user_line2.png",
            "type": "User Real Handwriting",
            "gt": "of everything, you'll be able to",
        },
        {
            "name": "printed_sample.png",
            "path": "experiments/trocr_poc/test_images/printed_sample.png",
            "type": "Machine-Printed Baseline",
            "gt": "Document Processing and Verification Unit",
        },
    ]

    # Verify test files exist
    for item in test_suite:
        if not os.path.exists(item["path"]):
            print(f"Error: test file missing at {item['path']}")
            return

    # 2. Load official pretrained Hugging Face artifacts
    model_id = "microsoft/trocr-base-handwritten"
    print(f"\nLoading official pretrained weights from Hugging Face: {model_id}...")
    t_start = time.time()
    processor = TrOCRProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device)
    model.eval()
    print(f"Model loaded successfully in {time.time() - t_start:.2f}s")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Inference loop
    print("\n" + "=" * 90)
    print(f"{'FILE':<20} | {'DIMS':<10} | {'GROUND TRUTH':<30} | {'PREDICTED':<25}")
    print("=" * 90)

    results = []
    for item in test_suite:
        img_path = item["path"]
        gt = item["gt"]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        t0 = time.time()
        error_msg = None
        pred_text = ""
        try:
            # Official expected preprocessing
            pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                generated_ids = model.generate(pixel_values, max_new_tokens=64)
            pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        except Exception as e:
            error_msg = str(e)
            print(f"Error on {item['name']}: {e}")

        latency = (time.time() - t0) * 1000.0
        cer, wer = compute_metrics(gt, pred_text) if not error_msg else (1.0, 1.0)

        record = {
            "file": item["name"],
            "type": item["type"],
            "dimensions": f"{w}x{h}",
            "ground_truth": gt,
            "predicted_text": pred_text,
            "cer": cer,
            "wer": wer,
            "latency_ms": round(latency, 1),
            "error": error_msg,
        }
        results.append(record)
        print(f"{item['name']:<20} | {f'{w}x{h}':<10} | {gt:<30} | {pred_text:<25}")

    print("=" * 90)

    # 4. Save results to experiments/trocr_poc/results.json
    out_json = "experiments/trocr_poc/results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved execution results to: {out_json}")


if __name__ == "__main__":
    main()
