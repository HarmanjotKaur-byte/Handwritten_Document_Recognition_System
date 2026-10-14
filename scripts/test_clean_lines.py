import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import torch
import numpy as np
from src.recognition.model import StandaloneLaiaCRNN

ckpt = torch.load('models/checkpoints/pylaia_iam_crnn.pt', map_location='cpu')
idx_to_sym = ckpt['idx_to_sym']
model = StandaloneLaiaCRNN(num_classes=len(idx_to_sym))
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

def transcribe(crop):
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    bg = cv2.medianBlur(cv2.dilate(gray, np.ones((5, 5), np.uint8)), 21)
    diff = 255 - cv2.absdiff(gray, bg)
    norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    h, w = norm.shape
    new_w = max(32, int(w * (128.0 / h)))
    if new_w % 8 != 0:
        new_w = ((new_w // 8) + 1) * 8
    resized = cv2.resize(norm, (new_w, 128), interpolation=cv2.INTER_AREA)
    tensor = 1.0 - (resized.astype(np.float32) / 255.0)
    inp = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        out = model(inp)
    probs = torch.softmax(out, dim=-1)
    preds = torch.argmax(probs, dim=-1)[0].numpy()
    chars, prev = [], None
    for p in preds:
        if p != prev:
            if p != 0 and p in idx_to_sym:
                chars.append(idx_to_sym[p])
            prev = p
    return ''.join(chars)

for i in [1, 2, 3]:
    im = cv2.imread(f'debug_line{i}.png')
    res = transcribe(im)
    print(f"Line {i}: '{res}'")
