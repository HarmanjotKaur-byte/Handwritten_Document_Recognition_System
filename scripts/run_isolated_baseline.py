"""Isolated Baseline Execution strictly using CodeMaster-AJ repository code."""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn

# 1. Dataset parsing & character dictionary reconstruction exactly as in notebook Block 1
TXT_PATH = "external/handwritten-text-recognition/iam_words/words.txt"
IMG_DIR = "external/handwritten-text-recognition/iam_words/words"
MODEL_SAVE_PATH = "external/handwritten-text-recognition/saved_models/best_model.pth"

def parse_iam_dataset(txt_path, img_dir):
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

all_samples, alphabet = parse_iam_dataset(TXT_PATH, IMG_DIR)
char_to_num = {char: idx + 1 for idx, char in enumerate(alphabet)}
num_to_char = {idx + 1: char for idx, char in enumerate(alphabet)}
num_classes = len(alphabet) + 1  # 77 classes

print(f"Dataset: {len(all_samples)} valid samples, {len(alphabet)} unique chars, {num_classes} classes.")

# 2. Exact Model Architecture from notebook Block 3
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1))
        )
        self.linear = nn.Linear(256 * 4, 128)
        self.lstm = nn.LSTM(128, 128, bidirectional=True, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        features = features.permute(0, 3, 1, 2)
        b, t, c, h = features.size()
        features = features.contiguous().view(b, t, c * h)
        seq_out = self.linear(features)
        seq_out, _ = self.lstm(seq_out)
        output_logits = self.fc(seq_out)
        return output_logits

device = torch.device("cpu")
model = CRNN(num_classes).to(device)

# Load checkpoint
print(f"Loading weights from {MODEL_SAVE_PATH}...")
state_dict = torch.load(MODEL_SAVE_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded successfully!")

# 3. Exact Inference Preprocessing and Decoder from notebook Block 5
def transcribe_custom_handwriting(user_image_path, model, char_map_decoder):
    img = cv2.imread(user_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return f"Target image file not found: {user_image_path}"
    h, w = img.shape
    f = min(128 / w, 32 / h)
    new_w, new_h = int(w * f), int(h * f)
    img_res = cv2.resize(img, (new_w, new_h))
    canvas = np.ones((32, 128), dtype=np.uint8) * 255
    canvas[0:new_h, 0:new_w] = img_res

    corners = [canvas[0, 0], canvas[0, -1], canvas[-1, 0], canvas[-1, -1]]
    if np.mean(corners) < 128:
        canvas = 255 - canvas

    tensor_input = torch.tensor(canvas.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(tensor_input)
        pred_indexes = torch.argmax(preds, dim=2).squeeze(0).cpu().numpy()

    decoded_string = ""
    last_seen_token = None
    for token in pred_indexes:
        if token != 0 and token != last_seen_token:
            decoded_string += char_map_decoder.get(token, "")
        last_seen_token = token
    return decoded_string

print("\n================ 1. REPO'S PROVIDED TEST IMAGE ================")
test_img_path = "external/handwritten-text-recognition/test_images/my_word.png"
pred_test = transcribe_custom_handwriting(test_img_path, model, num_to_char)
print(f"Image Path: {test_img_path}")
print(f"Expected Transcription: 'if' (per notebook cell 1 output)")
print(f"Predicted Transcription: '{pred_test}'")
print(f"Match: {pred_test == 'if'}")

print("\n================ 2. REPEATABILITY TEST (3 RUNS) ================")
runs = [transcribe_custom_handwriting(test_img_path, model, num_to_char) for _ in range(3)]
print(f"Run 1: '{runs[0]}'")
print(f"Run 2: '{runs[1]}'")
print(f"Run 3: '{runs[2]}'")
print(f"Consistent across repeated runs: {runs[0] == runs[1] == runs[2]}")

print("\n================ 3. IAM GROUND-TRUTH TEST SAMPLES ================")
# Test first 10 IAM samples and report exact ground-truth vs prediction
for idx in range(10):
    img_p, gt = all_samples[idx]
    pred = transcribe_custom_handwriting(img_p, model, num_to_char)
    status = "MATCH" if pred == gt else ("CASE-INSENSITIVE MATCH" if pred.lower() == gt.lower() else "DIFF")
    print(f"Sample {idx+1:2d} | Path: {img_p}")
    print(f"          GT: '{gt}' | Pred: '{pred}' | [{status}]")
