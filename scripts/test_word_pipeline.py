import sys
sys.path.insert(0, ".")
import cv2
from src.ocr.recognizer import HandwritingRecognizer
from src.preprocessing.image_processor import ImageProcessor, PreprocessingOptions
from src.preprocessing.segmenter import LineSegmenter

proc = ImageProcessor()
segm = LineSegmenter()
rec = HandwritingRecognizer(engine_type='trocr')

doc_img = cv2.imread('user_notebook_full.png')
prep = proc.process_page(doc_img, PreprocessingOptions())
desk = prep.deskewed if prep.deskewed is not None else prep.processed
bin_img, _ = proc.binarize(desk, 'otsu')
lines = segm.segment_lines(bin_img, desk)

res = rec.recognize_lines(lines, apply_lexicon=False)
print(f"Recognized lines count: {len(res.lines)}")
for l in res.lines:
    print(f"Line #{l.line_index + 1}: \"{l.text}\" ({len(l.words)} words)")
    for w in l.words[:3]:
        print(f"   Word: '{w.text}' -> bbox {w.bbox}")

print("\nFinal clean full text:")
print("-" * 50)
print(res.full_text)
print("-" * 50)
