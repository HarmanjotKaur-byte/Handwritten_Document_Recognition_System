import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2

class CharacterMapper:
    def __init__(self):
        chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?\'\"()-")
        self.chars = sorted(list(chars))
        self.char2idx = {c: i+1 for i, c in enumerate(self.chars)}
        self.idx2char = {i+1: c for i, c in enumerate(self.chars)}
        self.idx2char[0] = ''
        self.num_classes = len(self.chars) + 1

    def decode(self, indices):
        chars, prev = [], None
        for idx in indices:
            if idx != 0 and idx != prev and idx in self.idx2char:
                chars.append(self.idx2char[idx])
            prev = idx
        return ''.join(chars)

sys.modules['__main__'].CharacterMapper = CharacterMapper

class CRNN(nn.Module):
    def __init__(self, img_height=128, num_chars=80, hidden_size=256, num_layers=2):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, 2), nn.BatchNorm2d(512), nn.ReLU(),
        )
        self.map2seq = nn.Linear(512 * 7, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, bidirectional=True,
                           dropout=0.3 if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_chars + 1)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.permute(0, 3, 1, 2).reshape(b, w, c * h)
        seq = self.map2seq(conv)
        rnn_out, _ = self.rnn(seq)
        output = self.fc(rnn_out)
        return torch.nn.functional.log_softmax(output, dim=2)

pth = r'C:/Users/hp/.cache/huggingface/hub/models--ismatsamadov--handwriting-recognition-iam/snapshots/ff40343e51f3de7efb7c608fafcce6679197e3d1/best_model.pth'
ckpt = torch.load(pth, map_location='cpu', weights_only=False)
mapper = ckpt['char_mapper']
model = CRNN(img_height=128, num_chars=len(mapper.chars), hidden_size=256, num_layers=2)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

user_img_path = r'C:/Users/hp/.gemini/antigravity/brain/cbe2ec0b-00ab-4ab3-bf33-aeae435587f8/.user_uploaded/media_1791629926283_738d9692.png'
img = cv2.imread(user_img_path)

# Let us segment using a robust connected components & horizontal projection algorithm
# 1. Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Illumination compensation / background division
background = cv2.morphologyEx(gray, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)))
diff = cv2.absdiff(background, gray)
diff_norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# 3. Thresholding ink
_, binary = cv2.threshold(diff_norm, 40, 255, cv2.THRESH_BINARY)

# 4. Remove large solid objects (like the pen) and remove ruled horizontal lines
# Remove ruled horizontal lines:
h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)
binary_no_lines = cv2.subtract(binary, lines)

# Remove the pen: pen has very large connected components or blue color in original image
# Pen detection: blue color mask in HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([135, 255, 255])
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
# Dilate blue mask slightly
blue_mask_dilated = cv2.dilate(blue_mask, np.ones((5, 5), np.uint8), iterations=2)

# Also remove components that are too huge (pen body)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_no_lines)
clean_text_binary = np.zeros_like(binary_no_lines)

for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    bw = stats[i, cv2.CC_STAT_WIDTH]
    bh = stats[i, cv2.CC_STAT_HEIGHT]
    # Filter out pen body (giant area > 10000 or blue pen overlap)
    mask = (labels == i)
    if area > 12000:
        continue
    if np.sum(blue_mask_dilated[mask]) / (area * 255.0) > 0.4:
        continue
    # Keep handwriting strokes
    clean_text_binary[mask] = 255

# 5. Now project horizontally to find text line bands
# Dilate text horizontally
merge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 5))
merged_lines = cv2.dilate(clean_text_binary, merge_kernel, iterations=1)

# Find bounding rects of merged text lines
contours, _ = cv2.findContours(merged_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
boxes = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if w > 80 and h > 15: # Valid text line
        boxes.append((x, y, w, h))

# Sort boxes top to bottom
boxes.sort(key=lambda b: b[1])
print(f'Found {len(boxes)} text lines:')

for i, (x, y, w, h) in enumerate(boxes):
    # Expand slightly
    pad_y = 6
    pad_x = 10
    y1 = max(0, y - pad_y)
    y2 = min(img.shape[0], y + h + pad_y)
    x1 = max(0, x - pad_x)
    x2 = min(img.shape[1], x + w + pad_x)
    
    line_crop = img[y1:y2, x1:x2]
    pil_crop = Image.fromarray(cv2.cvtColor(line_crop, cv2.COLOR_BGR2RGB)).convert('L')
    
    # Preprocess for IAM CRNN
    lw, lh = pil_crop.size
    new_w = max(32, int(128 * (lw / lh)))
    pil_crop = pil_crop.resize((new_w, 128), Image.Resampling.LANCZOS)
    arr = np.array(pil_crop, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    t = torch.FloatTensor(arr).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        out = model(t)
        preds = torch.argmax(out, dim=2)[0].tolist()
        pred_text = mapper.decode(preds)
    
    print(f'Line {i+1} [y={y1}:{y2}, x={x1}:{x2}]: "{pred_text}"')
