import os
import sys
import torch
import torch.nn as nn
from PIL import Image
from datasets import load_dataset
import jiwer

sys.path.insert(0, ".")

from src.recognition.model import CRNNRecognitionModel
from src.recognition.vocabulary import Vocabulary
from src.recognition.decoder import GreedyCTCDecoder
from src.recognition.preprocessing import LineImagePreprocessor

print("=" * 60)
print("PHASE 3: TARGETED OVERFITTING VERIFICATION")
print("=" * 60)

print("[1] Loading 2 fixed IAM training lines...")
ds = load_dataset("Teklia/IAM-line", split="train")

samples = [ds[0], ds[1]]
for i, s in enumerate(samples):
    print(f"  Line {i}: \"{s['text']}\"")

texts = [s["text"] for s in samples]
vocab = Vocabulary.build_from_texts(texts, use_unk=False)
print(f"[2] Vocabulary size: {len(vocab)} classes (blank_index={vocab.BLANK_INDEX})")

preprocessor = LineImagePreprocessor(target_height=128)
decoder = GreedyCTCDecoder(vocabulary=vocab, blank_index=vocab.BLANK_INDEX)

model = CRNNRecognitionModel(
    num_classes=len(vocab),
    in_channels=1,
    lstm_hidden_size=128,
    lstm_num_layers=1,
    lstm_dropout=0.0,
    img_height=128,
)

criterion = nn.CTCLoss(blank=vocab.BLANK_INDEX, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

# Prepare each sample individually (variable widths, avoid large padding tensor)
sample_data = []
for s in samples:
    img_t = preprocessor.preprocess(s["image"]) # (1, 128, W)
    if img_t.ndim == 3:
        img_t = img_t.unsqueeze(0) # (1, 1, 128, W)
    w = img_t.shape[3]
    in_len = torch.tensor([max(1, (w // 4) - 1)], dtype=torch.long)
    tgt_enc = torch.tensor(vocab.encode(s["text"]), dtype=torch.long)
    tgt_len = torch.tensor([len(tgt_enc)], dtype=torch.long)
    sample_data.append((img_t, in_len, tgt_enc, tgt_len, s["text"]))

# Pre-test
model.eval()
print("\n--- Predictions BEFORE Overfitting (Random Weights) ---")
with torch.no_grad():
    for i, (img_t, in_len, tgt_enc, tgt_len, text) in enumerate(sample_data):
        init_out = model(img_t)
        pred_before = decoder.decode(init_out, input_lengths=in_len)[0][0]
        print(f"Sample {i}:")
        print(f"  GT:   \"{text}\"")
        print(f"  PRED: \"{pred_before}\"")

# Train loop (30 steps)
print("\n--- Training to Overfit ---")
model.train()
initial_loss = None
for step in range(1, 31):
    step_loss = 0.0
    optimizer.zero_grad()
    for img_t, in_len, tgt_enc, tgt_len, text in sample_data:
        logits = model(img_t)
        loss = criterion(logits.permute(1, 0, 2), tgt_enc, in_len, tgt_len)
        loss.backward()
        step_loss += loss.item()
    
    if initial_loss is None:
        initial_loss = step_loss
        
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    
    if step % 5 == 0 or step == 1:
        print(f"Step {step:2d}/30 | Combined CTC Loss: {step_loss:.4f}")

# Post-test
model.eval()
print("\n--- Predictions AFTER Overfitting ---")
all_gt = []
all_pred = []
with torch.no_grad():
    for i, (img_t, in_len, tgt_enc, tgt_len, text) in enumerate(sample_data):
        final_out = model(img_t)
        pred_after = decoder.decode(final_out, input_lengths=in_len)[0][0]
        cer = jiwer.cer(text, pred_after)
        all_gt.append(text)
        all_pred.append(pred_after)
        print(f"Sample {i}:")
        print(f"  GT:   \"{text}\"")
        print(f"  PRED: \"{pred_after}\" (CER: {cer:.1%})")

avg_cer = jiwer.cer(all_gt, all_pred)
print(f"\nInitial Loss: {initial_loss:.4f} -> Final Loss: {step_loss:.4f}")
print(f"Average CER on Overfit Set: {avg_cer:.1%}")

if avg_cer < 0.20:
    print("SUCCESS: The model architecture, input downsampling, and CTC loss successfully learned to memorize text!")
else:
    print("FAILURE: Model failed to overfit.")
