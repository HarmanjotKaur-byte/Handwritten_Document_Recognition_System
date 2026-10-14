"""Verification of CodeMaster-CRNN integration with WordSegmenter."""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from src.ocr.recognizer import HandwritingRecognizer
from src.preprocessing.segmenter import LineSegment

# 1. Initialize recognizer with codemaster engine
recognizer = HandwritingRecognizer(engine_type="codemaster")
diag = recognizer.get_status_diagnostics()
print("Recognizer Diagnostics:")
for k, v in diag.items():
    print(f"  {k}: {v}")

# 2. Test on a composite handwritten test strip
img_dir = "external/handwritten-text-recognition/iam_words/words"
sample1 = cv2.imread(f"{img_dir}/a01/a01-000u/a01-000u-00-00.png")  # 'A'
sample2 = cv2.imread(f"{img_dir}/a01/a01-000u/a01-000u-00-02.png")  # 'to'
sample3 = cv2.imread(f"{img_dir}/a01/a01-000u/a01-000u-01-01.png")  # 'any'

# Build a simulated line strip with whitespace gaps
line_h = 60
pad_gap = np.ones((line_h, 35, 3), dtype=np.uint8) * 255

def resize_to_h(im, h):
    ih, iw = im.shape[:2]
    nw = int(iw * (h / ih))
    return cv2.resize(im, (nw, h))

s1 = resize_to_h(sample1, line_h)
s2 = resize_to_h(sample2, line_h)
s3 = resize_to_h(sample3, line_h)

multi_word_line = np.hstack([s1, pad_gap, s2, pad_gap, s3])
cv2.imwrite("test_multi_word_strip.png", multi_word_line)
print(f"Created multi-word line strip of shape: {multi_word_line.shape}")

# 3. Recognize line
seg = LineSegment(
    index=0,
    crop=multi_word_line,
    x=0, y=0,
    w=multi_word_line.shape[1],
    h=multi_word_line.shape[0]
)

result = recognizer.recognize_lines([seg], apply_lexicon=False)
print(f"\nExpected Text: 'A to any'")
print(f"Recognized Text: '{result.full_text}'")
print(f"Line count: {len(result.lines)}")
