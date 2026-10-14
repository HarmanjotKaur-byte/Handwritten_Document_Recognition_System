import sys, os
from pathlib import Path
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_codemaster_model import transcribe

# Create a clean printed text image
img_pil = Image.new('RGB', (200, 50), color=(255, 255, 255))
d = ImageDraw.Draw(img_pil)
d.text((10, 15), "Invoice", fill=(0, 0, 0))
img_pil.save("external/test_printed_word.png")

res = transcribe("external/test_printed_word.png")
print("Transcribed printed word 'Invoice':", f"'{res}'")

# Test another printed word
img_pil2 = Image.new('RGB', (200, 50), color=(255, 255, 255))
d2 = ImageDraw.Draw(img_pil2)
d2.text((10, 15), "Patient", fill=(0, 0, 0))
img_pil2.save("external/test_printed_word2.png")
res2 = transcribe("external/test_printed_word2.png")
print("Transcribed printed word 'Patient':", f"'{res2}'")
