import os
import cv2
import numpy as np
import torch
import torch.nn as nn

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

repo = "external/handwritten-text-recognition"
alphabet = '!"#\'()*,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
num_to_char = {idx + 1: char for idx, char in enumerate(alphabet)}
num_classes = len(alphabet) + 1

model = CRNN(num_classes)
ckpt_path = os.path.join(repo, "saved_models/best_model.pth")
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt)
model.eval()
print("Model loaded successfully from:", ckpt_path)

def transcribe(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return f"Image not found: {img_path}"
    h, w = img.shape
    f = min(128.0 / w, 32.0 / h)
    new_w, new_h = max(1, int(w * f)), max(1, int(h * f))
    img_res = cv2.resize(img, (new_w, new_h))
    canvas = np.ones((32, 128), dtype=np.uint8) * 255
    canvas[0:new_h, 0:new_w] = img_res

    # Check shadows / inversion
    corners = [canvas[0, 0], canvas[0, -1], canvas[-1, 0], canvas[-1, -1]]
    if np.mean(corners) < 128:
        canvas = 255 - canvas

    tensor_input = torch.tensor(canvas.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        preds = model(tensor_input)
        pred_indexes = torch.argmax(preds, dim=2).squeeze(0).cpu().numpy()

    decoded_string = ""
    last_seen_token = None
    for token in pred_indexes:
        if token != 0 and token != last_seen_token:
            decoded_string += num_to_char.get(token, "")
        last_seen_token = token
    return decoded_string

my_word = os.path.join(repo, "test_images/my_word.png")
res = transcribe(my_word)
print(f"Prediction for my_word.png: '{res}'")
