import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from PIL import Image
from src.ocr.recognizer import HandwritingRecognizer
from src.preprocessing.image_processor import ImageProcessor, PreprocessingOptions
from src.preprocessing.segmenter import LineSegmenter

recognizer = HandwritingRecognizer(engine_type="pylaia")
im_path = r"C:/Users/hp/.gemini/antigravity/brain/cbe2ec0b-00ab-4ab3-bf33-aeae435587f8/.user_uploaded/media_1791632289142_7b8e8d94.png"
active_image = Image.open(im_path)

processor = ImageProcessor()
segmenter = LineSegmenter()
prep_options = PreprocessingOptions(apply_grayscale=True, apply_deskew=False)
prep_doc = processor.process_page(active_image, options=prep_options)
deskewed_source = prep_doc.deskewed if prep_doc.deskewed is not None else prep_doc.processed
bin_for_seg, _ = processor.binarize(deskewed_source, method="otsu")
segments = segmenter.segment_lines(bin_for_seg, deskewed_source)

print(f"Segmented {len(segments)} lines.")
print("Running recognizer with Lexicon Language Correction...")
res = recognizer.recognize_lines(segments, apply_lexicon=True)

print("\n--- FULL TEXT OUTPUT ---")
print(res.full_text)
print("------------------------\n")
for line in res.lines:
    print(f"Line {line.line_index}: \"{line.text}\" (conf: {line.confidence:.2f})")
