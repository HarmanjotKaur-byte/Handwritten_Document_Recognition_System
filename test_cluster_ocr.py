import cv2
import numpy as np
from src.recognition.predictor import HandwritingPredictor

p = HandwritingPredictor("models/checkpoints/iam_crnn_best.pt")
img = cv2.imread("exact_doc.png")

boxes = [
    (34, 29, 203, 21),
    (38, 46, 200, 21),
    (44, 62, 214, 26)
]

for idx, (x, y, w, h) in enumerate(boxes):
    crop = img[y:y+h, x:x+w]
    text, conf = p.recognize_line(crop)
    print(f"Line {idx+1}: '{text}' (conf: {conf:.1%})")
