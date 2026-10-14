import cv2
import torch
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recognition.model import CodeMasterCRNN

ckpt = torch.load('models/checkpoints/codemaster_crnn_trained.pth', map_location='cpu')
model = CodeMasterCRNN(num_classes=77)
model.load_state_dict(ckpt)
model.eval()

ALPHABET = "!\"#'()*,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
NUM_TO_CHAR = {idx + 1: char for idx, char in enumerate(ALPHABET)}

im = cv2.imread(r'C:/Users/hp/.gemini/antigravity/brain/cbe2ec0b-00ab-4ab3-bf33-aeae435587f8/.user_uploaded/media_1791632289142_7b8e8d94.png')
# Crop Line 1 (y=89, h=37, w=410, x from left to right)
line1 = im[89:126, 30:440]
cv2.imwrite('test_line1.png', line1)

h, w, _ = line1.shape
print(f'Line 1 dimensions: {w}x{h}')

def transcribe_crop(crop):
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ch, cw = crop.shape
    if ch == 0 or cw == 0:
        return ''
    f = min(128.0 / cw, 32.0 / ch)
    nw, nh = max(1, int(cw * f)), max(1, int(ch * f))
    res = cv2.resize(crop, (nw, nh))
    canvas = np.ones((32, 128), dtype=np.uint8) * 255
    canvas[0:nh, 0:nw] = res
    if np.mean([canvas[0, 0], canvas[0, -1], canvas[-1, 0], canvas[-1, -1]]) < 128:
        canvas = 255 - canvas
    t = torch.tensor(canvas.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        out = model(t)
        preds = torch.argmax(out, dim=2).squeeze(0).numpy()
    s = ''
    prev = None
    for p in preds:
        if p != 0 and p != prev:
            s += NUM_TO_CHAR.get(p, '')
        prev = p
    return s

print('Transcribing full line 1 as single block:')
print('Result:', transcribe_crop(line1))

print('\nTesting word slicing across line 1:')
step = w // 6
for idx in range(6):
    chunk = line1[:, idx * step : (idx + 1) * step]
    print(f'Chunk {idx + 1} ({idx * step}:{(idx + 1) * step}): {transcribe_crop(chunk)}')
