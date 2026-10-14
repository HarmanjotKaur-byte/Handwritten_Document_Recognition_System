import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from datasets import load_dataset
import numpy as np
import cv2

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
        c = self.conv(x)
        b, ch, h, w = c.shape
        c = c.permute(3, 0, 1, 2).reshape(w, b, ch * h)
        r, _ = self.rnn(c)
        logits = self.linear(r)
        return logits.permute(1, 0, 2)

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

ds = load_dataset('Teklia/IAM-line', split='validation')
print('Validation samples count:', len(ds))

for i in range(5):
    item = ds[i]
    pil_img = item['image'].convert('L')
    gt = item['text']
    arr = np.array(pil_img)
    
    # Preprocess
    h, w = arr.shape
    new_w = max(32, int(w * (128.0 / h)))
    if new_w % 8 != 0:
        new_w = ((new_w // 8) + 1) * 8
    resized = cv2.resize(arr, (new_w, 128), interpolation=cv2.INTER_AREA)
    tensor = 1.0 - (resized.astype(np.float32) / 255.0)
    inp = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        out = model(inp)
    pred = ctc_decode(out)
    print(f'Sample {i}:')
    print(f'  GT:   "{gt}"')
    print(f'  PRED: "{pred}"')
