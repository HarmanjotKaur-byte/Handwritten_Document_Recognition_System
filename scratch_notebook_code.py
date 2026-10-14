
# --- CODE CELL ---
# Install required packages
!pip install -q datasets transformers jiwer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
from PIL import Image
from tqdm import tqdm
from jiwer import cer, wer
import matplotlib.pyplot as plt

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Using device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# --- CODE CELL ---
class CRNN(nn.Module):
    """CNN-BiLSTM-CTC for Handwriting Recognition"""

    def __init__(self, img_height=128, num_chars=80, hidden_size=256, num_layers=2):
        super(CRNN, self).__init__()

        # CNN Feature Extractor
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

print("✓ Model architecture defined")

# --- CODE CELL ---
class CharacterMapper:
    def __init__(self):
        chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?\'"()-')
        self.chars = sorted(list(chars))
        self.char2idx = {c: i+1 for i, c in enumerate(self.chars)}
        self.idx2char = {i+1: c for i, c in enumerate(self.chars)}
        self.idx2char[0] = ''  # CTC blank
        self.num_classes = len(self.chars) + 1

    def encode(self, text):
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def decode(self, indices):
        chars, prev = [], None
        for idx in indices:
            if idx != 0 and idx != prev and idx in self.idx2char:
                chars.append(self.idx2char[idx])
            prev = idx
        return ''.join(chars)

char_mapper = CharacterMapper()
print(f"✓ Character mapper: {char_mapper.num_classes} classes")

# --- CODE CELL ---
class IAMDataset(Dataset):
    def __init__(self, split='train', img_height=128):
        self.img_height = img_height
        self.dataset = load_dataset("Teklia/IAM-line", split=split)
        print(f"  Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img = sample['image'].convert('L')
        text = sample['text']

        # Resize
        w, h = img.size
        new_w = int(self.img_height * (w / h))
        img = img.resize((new_w, self.img_height), Image.Resampling.LANCZOS)

        # Normalize
        img = np.array(img, dtype=np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = torch.FloatTensor(img).unsqueeze(0)

        target = torch.LongTensor(char_mapper.encode(text))
        return img, target, len(target), text

def collate_fn(batch):
    # Filter out samples with empty targets before processing the batch
    filtered_batch = [item for item in batch if item[1].numel() > 0]

    if not filtered_batch:
        # If all samples in the batch resulted in empty targets, return empty tensors
        # This batch will effectively be skipped by the DataLoader downstream without error.
        return {
            'images': torch.empty(0),
            'targets': torch.empty(0, dtype=torch.long),
            'target_lengths': torch.empty(0, dtype=torch.long),
            'input_lengths': torch.empty(0, dtype=torch.long),
            'texts': []
        }

    images, targets, target_lengths, texts = zip(*filtered_batch)

    max_w = max(img.shape[2] for img in images)
    b, h = len(images), images[0].shape[1]

    padded_imgs = torch.zeros(b, 1, h, max_w)
    input_lengths = []

    for i, img in enumerate(images):
        w = img.shape[2]
        padded_imgs[i, :, :, :w] = img
        # Ensure input_length is at least 1, as CTC expects non-zero length
        # and (w // 4) - 1 can be 0 or negative for very small images.
        input_lengths.append(max(1, (w // 4) - 1))

    return {
        'images': padded_imgs,
        'targets': torch.cat(targets), # Now all targets in 'targets' will be non-empty
        'target_lengths': torch.LongTensor(target_lengths),
        'input_lengths': torch.LongTensor(input_lengths),
        'texts': texts
    }

print("Loading datasets...")
train_dataset = IAMDataset('train')
val_dataset = IAMDataset('validation')

# Adjust batch_size to resolve OutOfMemoryError
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=2)

print(f"✓ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# --- CODE CELL ---
def decode_predictions(outputs, char_mapper):
    _, max_indices = torch.max(outputs, dim=2)
    return [char_mapper.decode(idx.cpu().numpy().tolist()) for idx in max_indices]

def compute_metrics(preds, truths):
    # Filter out empty ground truths as jiwer handles empty predictions but needs non-empty truths
    # Also, ensure predictions are not None
    valid_pairs = []
    for p, g in zip(preds, truths):
        if g: # Only include if ground truth is not empty
            valid_pairs.append((p if p else "", g)) # Use empty string for empty predictions

    if not valid_pairs:
        return 0.0, 0.0 # Return 0,0 if no valid pairs to compare (e.g., all ground truths were empty)

    preds_filtered, truths_filtered = zip(*valid_pairs)
    try:
        return cer(list(truths_filtered), list(preds_filtered)), wer(list(truths_filtered), list(preds_filtered))
    except: # Fallback for unexpected errors during metric computation
        return 0.0, 0.0

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}")

    for batch in pbar:
        images = batch['images'].to(device)
        targets = batch['targets'].to(device)
        input_lengths = batch['input_lengths']
        target_lengths = batch['target_lengths']

        outputs = model(images)
        outputs = outputs.permute(1, 0, 2)  # CTC format

        loss = criterion(outputs, targets, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_truths = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            images = batch['images'].to(device)
            targets = batch['targets'].to(device)
            input_lengths = batch['input_lengths']
            target_lengths = batch['target_lengths']
            texts = batch['texts']

            outputs = model(images)
            outputs_ctc = outputs.permute(1, 0, 2)
            loss = criterion(outputs_ctc, targets, input_lengths, target_lengths)
            total_loss += loss.item()

            preds = decode_predictions(outputs, char_mapper)
            all_preds.extend(preds)
            all_truths.extend(texts)

    avg_loss = total_loss / len(loader)
    cer_score, wer_score = compute_metrics(all_preds, all_truths)

    # Show examples
    print("\nExample predictions:")
    for i in range(min(3, len(all_preds))):
        print(f"  GT:   {all_truths[i]}")
        print(f"  Pred: {all_preds[i]}")

    return avg_loss, cer_score, wer_score

print("✓ Training functions ready")

# --- CODE CELL ---
if not os.path.exists('best_model.pth'):
    print("❌ Error: best_model.pth not found!")
    print("   Please either:")
    print("   1. Upload the model using the cell above, OR")
    print("   2. Run the training cells to generate the model")
else:
    # Load checkpoint - weights_only=False is safe here since this is your own checkpoint
    checkpoint = torch.load('best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Loaded best model (Epoch {checkpoint['epoch']}, CER: {checkpoint['val_cer']:.4f})")

    # Test on validation samples
    test_batch = next(iter(val_loader))

    with torch.no_grad():
        images = test_batch['images'].to(device)
        outputs = model(images)
        predictions = decode_predictions(outputs, char_mapper)

    # Visualize predictions
    fig, axes = plt.subplots(5, 1, figsize=(16, 15))

    for i in range(5):
        img = test_batch['images'][i, 0].cpu().numpy()
        img = (img * 0.5) + 0.5  # Denormalize

        gt = test_batch['texts'][i]
        pred = predictions[i]

        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"GT: {gt}\nPrediction: {pred}", fontsize=11, loc='left')
        axes[i].axis('off')

    plt.suptitle('Predictions on Validation Set', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150)
    plt.show()

    print("\n✓ Predictions saved to 'predictions.png'")

