"""Test TrOCR on handwritten line crops from user's notebook."""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

print("Loading TrOCR model (microsoft/trocr-small-handwritten)...")
t0 = time.time()
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
model.eval()
print(f"TrOCR loaded in {time.time() - t0:.2f}s")

user_p = r"C:/Users/hp/.gemini/antigravity/brain/cbe2ec0b-00ab-4ab3-bf33-aeae435587f8/.user_uploaded/media_1791632289142_7b8e8d94.png"
if not os.path.exists(user_p):
    print("User image not found!")
    sys.exit(1)

img = cv2.imread(user_p)
# User notebook lines:
# Line 1: If you can see the positive side
# Line 2: of everything you will be able to
# Line 3: live a rich and beautiful life.
line_crops = [
    ("Line 1", img[89:126, 30:440]),
    ("Line 2", img[116:150, 30:440]),
    ("Line 3", img[150:190, 30:440]),
]

print("\n--- Running TrOCR Inference on User Notebook Lines ---")
for name, crop in line_crops:
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)

    t_start = time.time()
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values

    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=64)

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    dt = time.time() - t_start
    print(f"  {name} ({dt:.2f}s): '{text}'")
