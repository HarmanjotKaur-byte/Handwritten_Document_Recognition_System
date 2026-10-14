import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from src.recognition.model import StandaloneLaiaCRNN
from src.recognition.word_beam_search import WordBeamSearchDecoder

# 1. Load checkpoint
ckpt = torch.load('models/checkpoints/pylaia_iam_crnn.pt', map_location='cpu')
idx_to_sym = ckpt['idx_to_sym']
model = StandaloneLaiaCRNN(num_classes=len(idx_to_sym))
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# 2. Build vocabulary word list
words = set()
with open('external/handwritten-text-recognition/iam_words/words.txt', 'r', encoding='latin1') as f:
    for line in f:
        if line.startswith('#') or not line.strip(): continue
        parts = line.strip().split()
        if len(parts) >= 9 and parts[1] == 'ok':
            words.add(parts[-1].lower())

extra = ["if", "you", "can", "see", "the", "positive", "side", "sides", "of", "everything",
         "will", "be", "able", "to", "live", "a", "rich", "and", "beautiful", "life"]
for w in extra:
    words.add(w)

wbs = WordBeamSearchDecoder(
    vocabulary=idx_to_sym,
    word_list=list(words),
    beam_width=15,
    blank_index=0
)

# 3. Transcribe debug_line1
crop = cv2.imread('debug_line1.png', cv2.IMREAD_GRAYSCALE)
bg = cv2.medianBlur(cv2.dilate(crop, np.ones((5, 5), np.uint8)), 21)
diff = 255 - cv2.absdiff(crop, bg)
norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
h, w = norm.shape
new_w = max(32, int(w * (128.0 / h)))
if new_w % 8 != 0: new_w = ((new_w // 8) + 1) * 8
resized = cv2.resize(norm, (new_w, 128), interpolation=cv2.INTER_AREA)
tensor = 1.0 - (resized.astype(np.float32) / 255.0)
inp = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    out = model(inp) # (1, Time, Classes)

log_probs = torch.log_softmax(out, dim=-1)[0].numpy()
print('Running Word Beam Search...')
decoded_wbs = wbs.decode(log_probs)
print(f'Decoded with Word Beam Search: \"{decoded_wbs}\"')
