import os
import cv2
import numpy as np
import torch
from test_codemaster_model import model, transcribe, num_to_char

# Let's test word segmentation on an image line
def segment_line_into_words(line_bgr_or_gray):
    if line_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(line_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = line_bgr_or_gray.copy()
        
    h, w = gray.shape
    # Binarize
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Dilate slightly vertically to connect letter strokes, but dilate gently horizontally
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # Find contours or vertical projection
    col_proj = np.sum(dilated > 0, axis=0)
    
    # Identify word bands
    words = []
    in_word = False
    start_x = 0
    min_space_width = max(6, int(h * 0.25))
    min_word_width = 8
    
    space_count = 0
    for x in range(w):
        if col_proj[x] > 0:
            if not in_word:
                in_word = True
                start_x = x
            space_count = 0
        else:
            if in_word:
                space_count += 1
                if space_count >= min_space_width:
                    end_x = x - space_count
                    if (end_x - start_x) >= min_word_width:
                        words.append((start_x, end_x))
                    in_word = False
                    space_count = 0
                    
    if in_word and (w - start_x) >= min_word_width:
        words.append((start_x, w))
        
    word_crops = []
    for sx, ex in words:
        # Add slight margin
        msx = max(0, sx - 2)
        mex = min(w, ex + 2)
        word_crop = gray[:, msx:mex]
        word_crops.append(word_crop)
        
    return word_crops

# Let's test on test_images/my_word.png
my_word_path = "external/handwritten-text-recognition/test_images/my_word.png"
im = cv2.imread(my_word_path, cv2.IMREAD_GRAYSCALE)
words = segment_line_into_words(im)
print(f"Segmented my_word.png into {len(words)} word(s)")
for idx, wc in enumerate(words):
    # Transcribe each word crop directly
    cv2.imwrite(f"external/temp_word_{idx}.png", wc)
    txt = transcribe(f"external/temp_word_{idx}.png")
    print(f" Word {idx}: '{txt}'")
