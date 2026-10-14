import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from src.recognition.predictor import get_predictor

p = get_predictor("models/checkpoints/pylaia_iam_crnn.pt")
print("Predictor is_pylaia:", p.is_pylaia)
print("Vocabulary sample:", [(k, p.vocabulary.idx_to_char[k]) for k in list(p.vocabulary.idx_to_char.keys())[:10]])

im = cv2.imread("debug_line1.png")
txt, conf = p.recognize_line(im)
print(f"debug_line1 recognize_line: '{txt}' (conf: {conf:.2f})")
