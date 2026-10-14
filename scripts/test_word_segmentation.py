"""Test projection profile based word segmentation."""

import cv2
import numpy as np


def segment_line_into_words(line_gray: np.ndarray):
    h, w = line_gray.shape[:2]
    if h < 8 or w < 16:
        return [(0, 0, w, h)]

    # 1. Otsu thresholding
    blurred = cv2.GaussianBlur(line_gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Vertical projection (column ink profile)
    proj = np.sum(binary > 0, axis=0)

    # 3. Identify runs of ink vs runs of space
    # Threshold for ink column: at least 1 pixel (or small percentage of height)
    ink_thresh = max(1, int(h * 0.05))
    is_ink = proj >= ink_thresh

    # Find continuous ink segments
    ink_segments = []
    in_ink = False
    start = 0
    for i, val in enumerate(is_ink):
        if val and not in_ink:
            start = i
            in_ink = True
        elif not val and in_ink:
            ink_segments.append((start, i))
            in_ink = False
    if in_ink:
        ink_segments.append((start, len(is_ink)))

    if not ink_segments:
        return [(0, 0, w, h)]

    # 4. Measure gap sizes between consecutive ink segments
    gaps = []
    for idx in range(len(ink_segments) - 1):
        gap_w = ink_segments[idx + 1][0] - ink_segments[idx][1]
        gaps.append(gap_w)

    # Intra-word gaps vs Inter-word gaps
    # Typically, inter-word gap is significantly wider (e.g. > 8px or > median gap * 2.0)
    if gaps:
        pos_gaps = [g for g in gaps if g > 1]
        if pos_gaps:
            med_gap = np.median(pos_gaps)
            # Gap threshold for a space between words
            word_gap_thresh = max(8, int(med_gap * 1.8), int(h * 0.25))
        else:
            word_gap_thresh = max(10, int(h * 0.3))
    else:
        word_gap_thresh = 15

    # 5. Merge segments separated by small gaps (< word_gap_thresh) into word blocks
    words = []
    curr_start = ink_segments[0][0]
    curr_end = ink_segments[0][1]

    for idx in range(len(ink_segments) - 1):
        next_start = ink_segments[idx + 1][0]
        next_end = ink_segments[idx + 1][1]
        gap = next_start - curr_end

        if gap < word_gap_thresh:
            curr_end = next_end
        else:
            # Word boundary
            words.append((curr_start, curr_end))
            curr_start = next_start
            curr_end = next_end
    words.append((curr_start, curr_end))

    # Convert word x-spans to bounding boxes
    boxes = []
    for x1, x2 in words:
        bw = x2 - x1
        if bw >= 4:
            # Expand slightly
            pad = 2
            bx1 = max(0, x1 - pad)
            bx2 = min(w, x2 + pad)
            boxes.append((bx1, 0, bx2 - bx1, h))

    return boxes if boxes else [(0, 0, w, h)]


if __name__ == "__main__":
    canvas = np.ones((60, 400), dtype=np.uint8) * 255
    cv2.putText(canvas, "Hello World IAM", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,), 2)
    boxes = segment_line_into_words(canvas)
    print("Boxes detected in 'Hello World IAM':", len(boxes))
    for i, b in enumerate(boxes):
        print(f"  Word {i+1}: {b}")
