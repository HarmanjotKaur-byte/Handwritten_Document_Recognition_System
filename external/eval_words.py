import os
import cv2
import torch
import numpy as np
from test_codemaster_model import model, transcribe, num_to_char, alphabet

repo = 'external/handwritten-text-recognition'
txt_path = os.path.join(repo, 'iam_words/words.txt')
img_dir = os.path.join(repo, 'iam_words/words')

samples = []
with open(txt_path, 'r', encoding='latin1') as f:
    for line in f:
        if line.startswith('#') or not line.strip(): continue
        parts = line.strip().split()
        if len(parts) >= 9 and parts[1] == 'ok':
            file_id = parts[0]
            word_text = parts[-1]
            id_parts = file_id.split('-')
            p = os.path.join(img_dir, id_parts[0], f'{id_parts[0]}-{id_parts[1]}', f'{file_id}.png')
            if os.path.exists(p):
                samples.append((p, word_text))

print('Total samples:', len(samples))
val_samples = samples[int(len(samples)*0.95):]
correct = 0
total = 20

print("\n--- Testing CodeMaster Pretrained Model on Unseen IAM Validation Words ---")
for i in range(total):
    img_p, gt = val_samples[i]
    pred = transcribe(img_p)
    is_corr = (pred.lower() == gt.lower())
    if is_corr:
        correct += 1
    print(f"Sample {i+1:2d}: GT='{gt}', PRED='{pred}' [{'MATCH' if is_corr else 'DIFF'}]")

print(f"\nExact Word Accuracy on 20 validation samples: {correct}/{total} ({correct/total:.1%})")
