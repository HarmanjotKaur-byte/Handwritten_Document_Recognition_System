import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from src.recognition.predictor import HandwritingPredictor

predictor = HandwritingPredictor("models/checkpoints/pylaia_iam_crnn.pt")
print("Predictor ready:", predictor.is_ready)

img_pil = Image.new('RGB', (400, 50), color=(255, 255, 255))
d = ImageDraw.Draw(img_pil)
d.text((10, 15), "INVOICE #1024 - TOTAL DUE: $450.00", fill=(0, 0, 0))

txt, conf = predictor.recognize_line(img_pil)
print("PyLaia on printed text:", f"'{txt}'", f"conf={conf:.1%}")
