import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from PIL import Image
from src.ocr.recognizer import HandwritingRecognizer
from src.preprocessing.image_processor import ImageProcessor, PreprocessingOptions
from src.preprocessing.segmenter import LineSegmenter

img_path = r'C:/Users/hp/.gemini/antigravity/brain/cbe2ec0b-00ab-4ab3-bf33-aeae435587f8/.user_uploaded/media_1791632289142_7b8e8d94.png'
im = Image.open(img_path)

proc = ImageProcessor()
opts = PreprocessingOptions(apply_grayscale=True, apply_denoise=True, apply_contrast_enhancement=True, apply_deskew=True)
prep = proc.process_page(im, opts)
deskewed = prep.deskewed if prep.deskewed is not None else prep.processed
bin_img, _ = proc.binarize(deskewed, method='otsu')

seg = LineSegmenter()
lines = seg.segment_lines(bin_img, deskewed)
print(f"Segmenter detected {len(lines)} lines:")
for l in lines:
    print(f"  Line #{l.index+1}: y={l.y}, h={l.h}, x={l.x}, w={l.w}")

rec = HandwritingRecognizer()
print(f"Recognizer status: {rec.iam_predictor.status_message if rec.iam_predictor else 'Baseline'}")

res = rec.recognize_lines(lines)
print("\n--- OCR Recognition Output ---")
for lr in res.lines:
    print(f"Line #{lr.line_index+1} (conf={lr.confidence:.1%}): '{lr.text}'")

print("\n--- Aggregated Document Text ---")
print(res.full_text)
