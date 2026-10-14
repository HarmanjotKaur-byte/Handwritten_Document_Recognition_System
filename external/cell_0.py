import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =====================================================================
# BLOCK 1: SET PATHS AND PARSE DATASET
# =====================================================================
TXT_PATH = "iam_words/words.txt"
IMG_DIR = "iam_words/words"
MODEL_SAVE_PATH = "saved_models/best_model.pth"

def parse_iam_dataset(txt_path, img_dir):
    samples = []
    chars = set()
    
    with open(txt_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
                
            parts = line.strip().split()
            if len(parts) < 9:
                continue
                
            # Filter status flag safely at index 1
            if parts[1] != "ok":
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
    print(f"Loaded {len(samples)} valid images. Found {len(alphabet)} unique characters.")
    return samples, alphabet

all_samples, alphabet = parse_iam_dataset(TXT_PATH, IMG_DIR)

char_to_num = {char: idx + 1 for idx, char in enumerate(alphabet)}
num_to_char = {idx + 1: char for idx, char in enumerate(alphabet)}
num_classes = len(alphabet) + 1  # +1 for blank token

# =====================================================================
# BLOCK 2: CUSTOM DATASET & MPS-COMPATIBLE COLLATOR
# =====================================================================
class IAMDataset(Dataset):
    def __init__(self, samples, char_to_num, img_w=128, img_h=32):
        self.samples = samples
        self.char_to_num = char_to_num
        self.img_w = img_w
        self.img_h = img_h
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, word = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
            
        h, w = img.shape
        fx = self.img_w / w
        fy = self.img_h / h
        f = min(fx, fy)
        new_w, new_h = int(w * f), int(h * f)
        img_resized = cv2.resize(img, (new_w, new_h))
        
        canvas = np.ones((self.img_h, self.img_w), dtype=np.uint8) * 255
        canvas[0:new_h, 0:new_w] = img_resized
        
        img_normalized = canvas.astype(np.float32) / 255.0
        img_tensor = np.expand_dims(img_normalized, axis=0) # Shape: (1, 32, 128)
        
        # Convert text characters directly to a dynamic, unpadded sequence list
        label_list = [self.char_to_num.get(char, 0) for char in word]
            
        return torch.tensor(img_tensor, dtype=torch.float32), label_list

# CRITICAL MPS FIX: Create a collator that flattens target label text into a continuous 1D array
def mps_ctc_collate(batch):
    images = []
    targets = []
    target_lengths = []
    
    for img, labels in batch:
        images.append(img)
        targets.extend(labels)
        target_lengths.append(len(labels))
        
    return (
        torch.stack(images, dim=0),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(target_lengths, dtype=torch.long)
    )

split_idx = int(len(all_samples) * 0.95)
train_dataset = IAMDataset(all_samples[:split_idx], char_to_num)
val_dataset = IAMDataset(all_samples[split_idx:], char_to_num)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=mps_ctc_collate)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=mps_ctc_collate)

# =====================================================================
# BLOCK 3: THE NEURAL NETWORK ENGINE (CRNN)
# =====================================================================
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

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = CRNN(num_classes).to(device)
print(f"Model mounted successfully on execution runtime engine: {device}")

# =====================================================================
# BLOCK 4: THE PIPELINE TRAINING ENGINE
# =====================================================================
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nStarting pipeline training loops (Running 5 fast test epochs)...")
for epoch in range(1, 6):
    model.train()
    running_loss = 0.0
    
    for images, labels, label_lengths in train_loader:
        images = images.to(device)
        optimizer.zero_grad()
        
        outputs = model(images)
        outputs_ctc = outputs.log_softmax(2).permute(1, 0, 2)
        
        batch_size = images.size(0)
        input_lengths = torch.full(size=(batch_size,), fill_value=32, dtype=torch.long, device=device)
        
        # Labels are now safe for the MPS engine's expectations
        loss = criterion(outputs_ctc, labels.to(device), input_lengths, label_lengths.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_size
        
    print(f"Epoch {epoch}/5 - Loss: {(running_loss / len(train_dataset)):.4f}")

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Weights successfully saved to: {MODEL_SAVE_PATH}")

# =====================================================================
# BLOCK 5: LIVE APPLICATION PREDICTION TESTER (INFERENCE)
# =====================================================================
def transcribe_custom_handwriting(user_image_path, model, char_map_decoder):
    img = cv2.imread(user_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "Target image file not found inside test_images directory."
        
    h, w = img.shape
    f = min(128 / w, 32 / h)
    new_w, new_h = int(w * f), int(h * f)
    img_res = cv2.resize(img, (new_w, new_h))
    
    canvas = np.ones((32, 128), dtype=np.uint8) * 255
    canvas[0:new_h, 0:new_w] = img_res
    
    tensor_input = torch.tensor(canvas.astype(np.float32)/255.0).unsqueeze(0).unsqueeze(0).to(device)
    
    model.eval()
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

print("\n--- Project Pipeline Complete ---")
print("To test a picture of your own handwriting:")
print("1. Drop your image into 'test_images/my_word.png'")
print("2. Run a new cell with: print(transcribe_custom_handwriting('test_images/my_word.png', model, num_to_char))")
