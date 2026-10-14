"""Test CodeMaster-CRNN with word-level line segmentation on the user's uploaded notebook image."""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.recognition.model import CodeMasterCRNN
from src.preprocessing.segmenter import LineSegmenter

ALPHABET = "!\"#'()*,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
NUM_TO_CHAR = {idx + 1: char for idx, char in enumerate(ALPHABET)}

def load_codemaster():
    ckpt_candidates = [
        "models/checkpoints/codemaster_crnn_trained.pth",
        "models/checkpoints/codemaster_crnn_finetuned.pth",
        "models/checkpoints/codemaster_crnn_best.pth",
    ]
    model_path = None
    for p in ckpt_candidates:
        if os.path.isfile(p):
            model_path = p
            break
    if not model_path:
        raise FileNotFoundError("No CodeMaster checkpoint found!")
    
    print(f"Loading checkpoint: {model_path}")
    model = CodeMasterCRNN(num_classes=77)
    ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt)
    model.eval()
    return model

def segment_line_words(line_gray: np.ndarray):
    h, w = line_gray.shape[:2]
    if h < 8 or w < 24:
        return [(0, 0, w, h)]
    
    # Otsu threshold
    blurred = cv2.GaussianBlur(line_gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    proj = np.sum(binary > 0, axis=0)
    ink_thresh = max(1, int(h * 0.05))
    is_ink = proj >= ink_thresh
    
    ink_segments = []
    in_ink = False
    start = 0
    for i, val in enumerate(is_ink):
        if val and not in_ink:
            start = i
            in_ink = True
        elif not val and in_ink:
            ink_segments.append((start, i))
            in_ink = False
    if in_ink:
        ink_segments.append((start, len(is_ink)))
        
    if not ink_segments:
        return [(0, 0, w, h)]
        
    gaps = []
    for idx in range(len(ink_segments) - 1):
        gap_w = ink_segments[idx + 1][0] - ink_segments[idx][1]
        gaps.append(gap_w)
        
    pos_gaps = [g for g in gaps if g > 1]
    if pos_gaps:
        med_gap = np.median(pos_gaps)
        word_gap_thresh = max(7, int(med_gap * 2.2))
    else:
        word_gap_thresh = 10
        
    words = []
    curr_start = ink_segments[0][0]
    curr_end = ink_segments[0][1]
    for idx in range(len(ink_segments) - 1):
        next_start = ink_segments[idx + 1][0]
        next_end = ink_segments[idx + 1][1]
        gap = next_start - curr_end
        if gap < word_gap_thresh:
            curr_end = next_end
        else:
            words.append((curr_start, curr_end))
            curr_start = next_start
            curr_end = next_end
    words.append((curr_start, curr_end))
    
    boxes = []
    for x1, x2 in words:
        bw = x2 - x1
        if bw >= 5:
            pad = 2
            bx1 = max(0, x1 - pad)
            bx2 = min(w, x2 + pad)
            boxes.append((bx1, 0, bx2 - bx1, h))
            
    return boxes if boxes else [(0, 0, w, h)]

def transcribe_word(word_gray: np.ndarray, model: nn.Module):
    h, w = word_gray.shape[:2]
    if h == 0 or w == 0:
        return "", 0.0
    f = min(128.0 / w, 32.0 / h)
    new_w, new_h = max(1, int(w * f)), max(1, int(h * f))
    img_res = cv2.resize(word_gray, (new_w, new_h))
    
    canvas = np.ones((32, 128), dtype=np.uint8) * 255
    canvas[0:new_h, 0:new_w] = img_res
    
    corners = [canvas[0, 0], canvas[0, -1], canvas[-1, 0], canvas[-1, -1]]
    if np.mean(corners) < 128:
        canvas = 255 - canvas
        
    tensor_input = torch.tensor(canvas.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        preds = model(tensor_input) # (1, 32, 77)
        probs = torch.softmax(preds, dim=2).squeeze(0)
        pred_indexes = torch.argmax(probs, dim=1).cpu().numpy()
        
    decoded_string = ""
    last_seen_token = None
    conf_scores = []
    for t_step, token in enumerate(pred_indexes):
        if token != 0 and token != last_seen_token:
            decoded_string += NUM_TO_CHAR.get(token, "")
            conf_scores.append(float(probs[t_step, token].item()))
        last_seen_token = token
        
    conf = float(np.mean(conf_scores)) if conf_scores else 0.0
    return decoded_string, conf

if __name__ == "__main__":
    user_img_path = r"C:\Users\hp\.gemini\antigravity\brain\cbe2ec0b-00ab-4ab3-bf33-aeae435587f8\.user_uploaded\media_1791632289142_7b8e8d94.png"
    if not os.path.exists(user_img_path):
        print("User image not found at path:", user_img_path)
        exit(1)
        
    img = cv2.imread(user_img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    from src.preprocessing.image_processor import ImageProcessor
    processor = ImageProcessor()
    binary, _ = processor.binarize(gray)
    
    segmenter = LineSegmenter()
    segments = segmenter.segment_lines(binary, gray)
    print(f"Segmented {len(segments)} lines from user notebook image.")
    
    model = load_codemaster()
    
    print("\n--- Transcription Results Line by Line ---")
    for i, seg in enumerate(segments[:10]):
        word_boxes = segment_line_words(seg.crop)
        words = []
        for wb in word_boxes:
            wx, wy, ww, wh = wb
            wc = seg.crop[wy:wy+wh, wx:wx+ww]
            text, conf = transcribe_word(wc, model)
            if text:
                words.append(text)
        line_text = " ".join(words)
        print(f"Line {i+1} ({len(word_boxes)} words): '{line_text}'")
