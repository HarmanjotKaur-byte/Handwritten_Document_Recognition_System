import os
import sys
import time
import torch
import torch.nn as nn
from datasets import load_dataset
from PIL import Image
import numpy as np

# Add project root
sys.path.insert(0, ".")

from src.recognition.model import CRNNRecognitionModel
from src.recognition.vocabulary import Vocabulary
from src.recognition.decoder import GreedyCTCDecoder
from src.recognition.preprocessing import LineImagePreprocessor

print("=" * 60)
print("PHASE 3: SMALL OVERFITTING EXPERIMENT ON FIXED IAM SAMPLES")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 1. Load IAM train split
print("\n[1] Loading 5 fixed IAM training samples...")
ds = load_dataset("Teklia/IAM-line", split="train")

NUM_SAMPLES = 5
fixed_samples = []
for i in range(NUM_SAMPLES):
    item = ds[i]
    fixed_samples.append({
        "id": i,
        "image": item["image"],
        "text": item["text"]
    })
    print(f"  Sample {i}: \"{item['text']}\" (size: {item['image'].size})")

# 2. Build or load Vocabulary
train_texts = [s["text"] for s in fixed_samples]
# Include characters needed for these samples plus spaces & punctuation
vocab = Vocabulary.build_from_texts(train_texts, use_unk=False)
print(f"\n[2] Built minimal vocabulary: {len(vocab)} classes (blank_index={vocab.BLANK_INDEX})")

preprocessor = LineImagePreprocessor(target_height=128, min_width=32, invert_colors=False)
decoder = GreedyCTCDecoder(vocabulary=vocab, blank_index=vocab.BLANK_INDEX)

# 3. Initialize fresh CRNN model
model = CRNNRecognitionModel(
    num_classes=len(vocab),
    in_channels=1,
    lstm_hidden_size=128,
    lstm_num_layers=2,
    lstm_dropout=0.0,
    img_height=128
).to(device)

criterion = nn.CTCLoss(blank=vocab.BLANK_INDEX, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Preprocess the 5 images into batch
tensors = []
targets = []
target_lengths = []
input_lengths = []

for s in fixed_samples:
    t = preprocessor.preprocess(s["image"]) # (1, 128, W)
    tensors.append(t)
    encoded = vocab.encode(s["text"])
    targets.extend(encoded)
    target_lengths.append(len(encoded))
    # Downsampling in our CNN: (W // 4) - 1
    input_lengths.append(max(1, (t.shape[2] // 4) - 1))

max_w = max(t.shape[2] for t in tensors)
# Ensure multiple of 4
if max_w % 4 != 0:
    max_w += (4 - (max_w % 4))

b = len(tensors)
batch_imgs = torch.zeros((b, 1, 128, max_w), dtype=torch.float32)
for i, t in enumerate(tensors):
    batch_imgs[i, :, :, :t.shape[2]] = t

batch_imgs = batch_imgs.to(device)
targets_tensor = torch.tensor(targets, dtype=torch.long).to(device)
input_lens_tensor = torch.tensor(input_lengths, dtype=torch.long).to(device)
target_lens_tensor = torch.tensor(target_lengths, dtype=torch.long).to(device)

print(f"\nBatch shape: {batch_imgs.shape}")
print(f"Target lengths: {target_lengths}")
print(f"Input lengths: {input_lengths}")

# Check predictions BEFORE training
model.eval()
with torch.no_grad():
    init_logits = model(batch_imgs) # (B, Time, C)
    init_preds = decoder.decode(init_logits, input_lengths=input_lens_tensor)

print("\n--- Predictions BEFORE Overfitting (Random Weights) ---")
for i in range(NUM_SAMPLES):
    print(f"Sample {i}:")
    print(f"  GT:   \"{fixed_samples[i]['text']}\"")
    print(f"  PRED: \"{init_preds[i][0]}\"")

# 4. Train loop for 80 steps
print("\n--- Training to Overfit ---")
model.train()
initial_loss = None
final_loss = None

for epoch in range(1, 91):
    optimizer.zero_grad()
    logits = model(batch_imgs) # (B, Time, C)
    
    # CTCLoss expects log_probs of shape (Time, Batch, C)
    log_probs = logits.permute(1, 0, 2)
    loss = criterion(log_probs, targets_tensor, input_lens_tensor, target_lens_tensor)
    
    if initial_loss is None:
        initial_loss = loss.item()
        
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    
    final_loss = loss.item()
    if epoch % 10 == 0 or epoch == 1:
        print(f"Step {epoch:2d}/90 | CTC Loss: {loss.item():.4f}")

# 5. Check predictions AFTER training
model.eval()
with torch.no_grad():
    final_logits = model(batch_imgs)
    final_preds = decoder.decode(final_logits, input_lengths=input_lens_tensor)

print("\n--- Predictions AFTER Overfitting ---")
import jiwer
all_gt = []
all_pred = []
for i in range(NUM_SAMPLES):
    gt = fixed_samples[i]["text"]
    pred = final_preds[i][0]
    all_gt.append(gt)
    all_pred.append(pred)
    cer = jiwer.cer(gt, pred)
    print(f"Sample {i}:")
    print(f"  GT:   \"{gt}\"")
    print(f"  PRED: \"{pred}\" (CER: {cer:.1%})")

avg_cer = jiwer.cer(all_gt, all_pred)
print(f"\nInitial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")
print(f"Average CER on Overfit Set: {avg_cer:.1%}")

if avg_cer < 0.20:
    print("SUCCESS: The model architecture, input downsampling, and CTC loss successfully learned to memorize text!")
else:
    print("FAILURE: Model failed to overfit. Diagnostic required.")
