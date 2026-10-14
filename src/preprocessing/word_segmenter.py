"""Word segmentation module for slicing text line strips into isolated words.

Enables word-level recognition models (like CodeMaster-CRNN) to accurately
transcribe full-sentence line crops without character compression artifacts.
Handles notebook ruled lines, varied cursive spacing, and lighting gradients.
"""

from typing import List, Tuple
import cv2
import numpy as np


class WordSegmenter:
    """Segments horizontal line images into isolated word bounding boxes."""

    def __init__(
        self,
        min_word_width: int = 10,
        min_word_height: int = 8,
        min_gap_ratio: float = 1.6,
    ):
        """Initialize WordSegmenter.

        Args:
            min_word_width: Minimum pixel width for a valid word candidate.
            min_word_height: Minimum pixel height for a valid word candidate.
            min_gap_ratio: Ratio above baseline whitespace gap to trigger word boundary.
        """
        self.min_word_width = min_word_width
        self.min_word_height = min_word_height
        self.min_gap_ratio = min_gap_ratio

    def _split_oversized_box(
        self, box: Tuple[int, int, int, int], binary: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """Split a candidate word box if its aspect ratio indicates merged words."""
        x, y, w, h = box
        # If the word box is not excessively wide, keep it intact
        if (w / max(1, h)) <= 2.5 or w < 65:
            return [box]

        strip = binary[y : y + h, x : x + w]
        col_proj = np.sum(strip > 0, axis=0)
        # Search the middle 60% of the box for the deepest valley
        m_start = int(w * 0.25)
        m_end = int(w * 0.75)
        if m_end <= m_start:
            return [box]

        valley_local = m_start + int(np.argmin(col_proj[m_start:m_end]))
        # Split into two sub-boxes
        b1 = (x, y, valley_local, h)
        b2 = (x + valley_local, y, w - valley_local, h)

        res = []
        for sub_b in [b1, b2]:
            if sub_b[2] >= self.min_word_width:
                res.extend(self._split_oversized_box(sub_b, binary))
        return res if res else [box]

    def segment_words(self, line_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Segment a line image into word bounding boxes ordered left-to-right.

        Args:
            line_image: Grayscale or BGR line crop.

        Returns:
            List of bounding boxes: [(x, y, w, h), ...] ordered left-to-right.
        """
        if line_image is None or line_image.size == 0:
            return []

        if line_image.ndim == 3:
            gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = line_image.copy()

        h, w = gray.shape[:2]
        if h < self.min_word_height or w < self.min_word_width:
            return [(0, 0, w, h)]

        # 1. Illumination normalization & Otsu binarization
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # 2. Suppress horizontal ruled notebook lines
        # Ruled lines are thin horizontal lines spanning at least 35 pixels
        hline_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
        hlines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hline_kernel)
        clean_binary = cv2.subtract(binary, hlines)

        # 3. Vertical projection profile (column-wise ink count)
        col_proj = np.sum(clean_binary > 0, axis=0)
        non_zero_proj = col_proj[col_proj > 0]
        p90_ink = float(np.percentile(non_zero_proj, 85)) if len(non_zero_proj) > 0 else 1.0

        # Treat columns with ink <= 12% of peak ink as gap valleys
        valley_threshold = max(1, int(p90_ink * 0.12))
        is_ink = col_proj > valley_threshold

        # 4. Locate contiguous ink segments
        ink_segments: List[Tuple[int, int]] = []
        in_segment = False
        start_x = 0

        for x, active in enumerate(is_ink):
            if active and not in_segment:
                start_x = x
                in_segment = True
            elif not active and in_segment:
                ink_segments.append((start_x, x))
                in_segment = False

        if in_segment:
            ink_segments.append((start_x, len(is_ink)))

        # 5. Check if projection found clean word divisions
        word_gap_threshold = max(8, int(h * 0.16))
        words: List[Tuple[int, int]] = []

        if len(ink_segments) > 1:
            curr_start = ink_segments[0][0]
            curr_end = ink_segments[0][1]

            for i in range(len(ink_segments) - 1):
                next_start = ink_segments[i + 1][0]
                next_end = ink_segments[i + 1][1]
                gap = next_start - curr_end

                if gap < word_gap_threshold:
                    curr_end = next_end
                else:
                    words.append((curr_start, curr_end))
                    curr_start = next_start
                    curr_end = next_end

            words.append((curr_start, curr_end))

        # 6. Fallback to morphological contour clustering if projection found <= 1 segment on wide crop
        if len(words) <= 1 and (w / max(1, h)) > 2.8:
            k_w = max(4, int(h * 0.22))
            k_h = max(3, int(h * 0.40))
            dil_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))
            dilated = cv2.dilate(clean_binary, dil_kernel, iterations=1)
            contours, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            raw_boxes = [cv2.boundingRect(c) for c in contours]
            filtered = [
                b for b in raw_boxes if b[2] >= self.min_word_width and b[3] >= 5
            ]
            if filtered:
                sorted_boxes = sorted(filtered, key=lambda b: b[0])
                words = [(b[0], b[0] + b[2]) for b in sorted_boxes]

        if not words:
            words = [(0, w)]

        # 7. Convert word spans into bounding boxes with padding
        boxes: List[Tuple[int, int, int, int]] = []
        pad = 3
        for x1, x2 in words:
            bw = x2 - x1
            if bw >= self.min_word_width:
                bx1 = max(0, x1 - pad)
                bx2 = min(w, x2 + pad)
                boxes.append((bx1, 0, bx2 - bx1, h))

        # 8. Bisect any remaining oversized boxes that contain fused words
        final_boxes: List[Tuple[int, int, int, int]] = []
        for box in boxes:
            final_boxes.extend(self._split_oversized_box(box, clean_binary))

        return final_boxes if final_boxes else [(0, 0, w, h)]
