"""Phase 7 Multi-Sample Comprehensive Standalone Benchmark Evaluation.

Evaluates CodeMaster-CRNN on:
1. Labeled IAM Training sample
2. Labeled IAM Unseen Validation sample
3. Blank / Noise image
4. Printed text sample
5. User notebook handwriting sample (media_1791632289142_7b8e8d94.png)
"""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ocr.recognizer import HandwritingRecognizer
from src.preprocessing.image_processor import ImageProcessor
from src.preprocessing.segmenter import LineSegment, LineSegmenter
from src.recognition.predictor import get_predictor


def compute_edit_distance(s1: str, s2: str) -> int:
    d = np.zeros((len(s1) + 1, len(s2) + 1), dtype=int)
    for i in range(len(s1) + 1):
        d[i, 0] = i
    for j in range(len(s2) + 1):
        d[0, j] = j
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = 1 + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return int(d[len(s1), len(s2)])


def compute_cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return compute_edit_distance(ref.lower(), hyp.lower()) / max(1, len(ref))


def compute_wer(ref: str, hyp: str) -> float:
    r_words = ref.lower().split()
    h_words = hyp.lower().split()
    if not r_words:
        return 0.0 if not h_words else 1.0
    return compute_edit_distance(r_words, h_words) / max(1, len(r_words))


def run_benchmark():
    predictor = get_predictor("models/checkpoints/codemaster_crnn_trained.pth")
    recognizer = HandwritingRecognizer()

    results = {}

    # -------------------------------------------------------------
    # 1. IAM Training Sample
    # -------------------------------------------------------------
    print("--- 1. Evaluating IAM Training Sample ---")
    train_img_path = "external/handwritten-text-recognition/test_images/my_word.png"
    # In test_images/my_word.png, ground truth is 'if'
    gt_train = "if"
    t0 = time.time()
    pred_train, conf_train = predictor.recognize_line(cv2.imread(train_img_path, cv2.IMREAD_GRAYSCALE))
    t_train = time.time() - t0
    cer_train = compute_cer(gt_train, pred_train)
    wer_train = compute_wer(gt_train, pred_train)
    print(f"  GT: '{gt_train}' | Pred: '{pred_train}' | Conf: {conf_train:.2f} | CER: {cer_train:.1%} | Time: {t_train*1000:.1f}ms")
    results["iam_train_sample"] = {
        "ground_truth": gt_train,
        "prediction": pred_train,
        "confidence": conf_train,
        "cer": cer_train,
        "wer": wer_train,
        "latency_ms": t_train * 1000,
    }

    # -------------------------------------------------------------
    # 2. IAM Unseen Validation Samples (Batch of 10)
    # -------------------------------------------------------------
    print("\n--- 2. Evaluating IAM Unseen Validation Samples ---")
    txt_path = "external/handwritten-text-recognition/iam_words/words.txt"
    img_dir = "external/handwritten-text-recognition/iam_words/words"
    val_samples = []
    with open(txt_path, "r", encoding="latin1") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 9 and parts[1] == "ok":
                fid = parts[0]
                wt = parts[-1]
                id_parts = fid.split("-")
                p = os.path.join(img_dir, id_parts[0], f"{id_parts[0]}-{id_parts[1]}", f"{fid}.png")
                if os.path.exists(p):
                    val_samples.append((p, wt))

    split = int(len(val_samples) * 0.95)
    unseen_set = val_samples[split : split + 10]
    val_cers, val_wers, val_times = [], [], []
    val_records = []
    for p, gt in unseen_set:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        t0 = time.time()
        pred, conf = predictor.recognize_line(img)
        dt = time.time() - t0
        c = compute_cer(gt, pred)
        w = compute_wer(gt, pred)
        val_cers.append(c)
        val_wers.append(w)
        val_times.append(dt * 1000)
        val_records.append({"gt": gt, "pred": pred, "cer": c, "match": (gt.lower() == pred.lower())})
        print(f"  GT: '{gt:12s}' | Pred: '{pred:12s}' | {'MATCH' if gt.lower() == pred.lower() else 'DIFF'} | CER: {c:.1%}")

    mean_val_cer = float(np.mean(val_cers))
    mean_val_wer = float(np.mean(val_wers))
    mean_val_time = float(np.mean(val_times))
    print(f"  --> Mean Val CER: {mean_val_cer:.1%}, Exact Match Rate: {np.mean([r['match'] for r in val_records]):.1%}")
    results["iam_validation_samples"] = {
        "mean_cer": mean_val_cer,
        "mean_wer": mean_val_wer,
        "mean_latency_ms": mean_val_time,
        "exact_match_rate": float(np.mean([r["match"] for r in val_records])),
        "samples": val_records,
    }

    # -------------------------------------------------------------
    # 3. Blank Image Test
    # -------------------------------------------------------------
    print("\n--- 3. Evaluating Blank / Noise Image ---")
    blank = np.ones((60, 200), dtype=np.uint8) * 255
    t0 = time.time()
    pred_blank, conf_blank = predictor.recognize_line(blank)
    t_blank = time.time() - t0
    print(f"  Input: Blank white canvas | Pred: '{pred_blank}' | Conf: {conf_blank:.2f} | Time: {t_blank*1000:.1f}ms")
    results["blank_image"] = {
        "prediction": pred_blank,
        "confidence": conf_blank,
        "hallucination": bool(len(pred_blank) > 0),
        "latency_ms": t_blank * 1000,
    }

    # -------------------------------------------------------------
    # 4. Printed Machine Text Test
    # -------------------------------------------------------------
    print("\n--- 4. Evaluating Printed Machine Text ---")
    printed_img = np.ones((60, 250), dtype=np.uint8) * 255
    cv2.putText(printed_img, "INVOICE", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,), 2)
    t0 = time.time()
    pred_printed, conf_printed = predictor.recognize_line(printed_img)
    t_printed = time.time() - t0
    gt_printed = "INVOICE"
    cer_printed = compute_cer(gt_printed, pred_printed)
    print(f"  GT: '{gt_printed}' | Pred: '{pred_printed}' | CER: {cer_printed:.1%} | Time: {t_printed*1000:.1f}ms")
    results["printed_text"] = {
        "ground_truth": gt_printed,
        "prediction": pred_printed,
        "cer": cer_printed,
        "latency_ms": t_printed * 1000,
    }

    # -------------------------------------------------------------
    # 5. User Notebook Image Test
    # -------------------------------------------------------------
    print("\n--- 5. Evaluating User Notebook Handwriting Sample ---")
    user_p = r"C:/Users/hp/.gemini/antigravity/brain/cbe2ec0b-00ab-4ab3-bf33-aeae435587f8/.user_uploaded/media_1791632289142_7b8e8d94.png"
    if os.path.exists(user_p):
        u_img = cv2.imread(user_p)
        u_gray = cv2.cvtColor(u_img, cv2.COLOR_BGR2GRAY)
        t0 = time.time()
        # Segment and recognize
        line_crops = [
            u_gray[89:126, 30:440],   # Line 1: If you can see the positive side
            u_gray[116:150, 30:440],  # Line 2: of everything you will be able to
            u_gray[150:190, 30:440],  # Line 3: live a rich and beautiful life.
        ]
        user_preds = []
        for l_idx, crop in enumerate(line_crops):
            l_pred, l_conf = predictor.recognize_line(crop)
            user_preds.append({"line": l_idx + 1, "text": l_pred, "conf": l_conf})
            print(f"  Line {l_idx+1}: '{l_pred}' (conf: {l_conf:.2f})")
        t_user = time.time() - t0
        full_text = "\n".join(r["text"] for r in user_preds)
        results["user_notebook"] = {
            "prediction": full_text,
            "lines": user_preds,
            "latency_ms": t_user * 1000,
        }

    # Save benchmark report to JSON
    report_path = "models/checkpoints/phase7_benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPhase 7 Benchmark evaluation saved to {report_path}")


if __name__ == "__main__":
    run_benchmark()
