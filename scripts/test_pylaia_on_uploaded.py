import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
import torch
import torch.nn as nn
import numpy as np

# Load checkpoint
ckpt = torch.load('models/checkpoints/pylaia_iam_crnn.pt', map_location='cpu')
idx_to_sym = ckpt['idx_to_sym']

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, pool):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.pool = nn.MaxPool2d(pool) if pool[0] > 0 else nn.Identity()

    def forward(self, x):
        return self.pool(self.act(self.batchnorm(self.conv(x))))

class PyLaiaCRNN(nn.Module):
    def __init__(self, num_classes=81):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(1, 12, (2, 2)),
            ConvBlock(12, 24, (2, 2)),
            ConvBlock(24, 48, (0, 0)),
            ConvBlock(48, 48, (2, 2)),
        )
        self.rnn = nn.LSTM(768, 256, num_layers=3, bidirectional=True, dropout=0.5)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        c = self.conv(x)  # (B, 48, 16, W')
        b, ch, h, w = c.shape
        c = c.permute(3, 0, 1, 2).reshape(w, b, ch * h)  # (Time, Batch, 768)
        r, _ = self.rnn(c)  # (Time, Batch, 512)
        logits = self.linear(r)  # (Time, Batch, 81)
        return logits.permute(1, 0, 2)  # (Batch, Time, 81)

model = PyLaiaCRNN()
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

def ctc_decode(logits):
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)[0].numpy()
    chars = []
    prev = None
    for p in preds:
        if p != prev:
            if p != 0 and p in idx_to_sym:
                chars.append(idx_to_sym[p])
            prev = p
    return ''.join(chars)

def preprocess_line(crop_bgr):
    if crop_bgr.ndim == 3:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop_bgr
    
    # Illumination compensation via morphological division
    bg = cv2.dilate(gray, np.ones((5, 5), np.uint8))
    bg = cv2.medianBlur(bg, 21)
    diff = 255 - cv2.absdiff(gray, bg)
    norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Resize keeping aspect ratio, target height 128
    h, w = norm.shape
    new_w = max(32, int(w * (128.0 / h)))
    if new_w % 8 != 0:
        new_w = ((new_w // 8) + 1) * 8
    resized = cv2.resize(norm, (new_w, 128), interpolation=cv2.INTER_AREA)
    
    # Invert normalized: ink=1.0, background=0.0
    tensor = 1.0 - (resized.astype(np.float32) / 255.0)
    return torch.tensor(tensor, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

im_path = r'C:/Users/hp/.gemini/antigravity/brain/cbe2ec0b-00ab-4ab3-bf33-aeae435587f8/.user_uploaded/media_1791632289142_7b8e8d94.png'
im = cv2.imread(im_path)
print(f"Loaded image {im_path}, shape={im.shape}")

# Let's inspect projection profile to find the actual lines
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# Otsu
_, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
row_sums = np.sum(bin_inv > 0, axis=1)

# Find peaks and valleys
print("\nScanning for line bands in user image:")
# Let's test with LineSegmenter
from src.preprocessing.segmenter import LineSegmenter
seg = LineSegmenter()
lines = seg.segment_lines(bin_inv, im)
print(f"Segmenter found {len(lines)} lines:")
for idx, line in enumerate(lines):
    inp = preprocess_line(line.crop)
    with torch.no_grad():
        out = model(inp)
    txt = ctc_decode(out)
    print(f"  Line {idx} (y={line.y}, h={line.h}, w={line.w}): '{txt}'")
