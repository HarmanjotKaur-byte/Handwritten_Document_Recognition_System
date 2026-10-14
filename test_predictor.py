from src.recognition.predictor import HandwritingPredictor
from PIL import Image

p = HandwritingPredictor("models/checkpoints/iam_crnn_best.pt")
print("Predictor ready:", p.is_ready, "status:", p.status_message)

for img_name in ["line1.png", "line2.png", "line3.png"]:
    img = Image.open(img_name)
    text, conf = p.recognize_line(img)
    print(f"{img_name} -> '{text}' (conf: {conf:.1%})")
