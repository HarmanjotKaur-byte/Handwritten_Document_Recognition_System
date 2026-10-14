import cv2
import sys
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

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

for name in ['line1.png', 'line2.png', 'line3.png']:
    line = cv2.imread(name)
    pil_img = Image.fromarray(cv2.cvtColor(line, cv2.COLOR_BGR2RGB)).convert('L')
    w, h = pil_img.size
    new_w = max(64, int(128 * (w / h)))
    pil_img = pil_img.resize((new_w, 128), Image.Resampling.LANCZOS)
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    t = torch.FloatTensor(arr).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        out = model(t)
        preds = torch.argmax(out, dim=2)[0].tolist()
        pred_text = mapper.decode(preds)
    print(f'{name} -> "{pred_text}"')
