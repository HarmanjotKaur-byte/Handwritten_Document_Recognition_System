"""Line and region segmentation for handwritten documents.

Segments preprocessed pages into individual line strips via horizontal projection
profiling, recursive valley splitting, and morphological bounding box analysis.
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class LineSegment:
    """Represents a segmented line crop and its bounding coordinates."""

    index: int
    x: int
    y: int
    w: int
    h: int
    crop: np.ndarray  # Cropped line image


class LineSegmenter:
    """Extracts text line strips from binarized document images."""

    def __init__(
        self,
        min_line_height: int = 14,
        min_line_width: int = 30,
        projection_threshold_ratio: float = 0.08,
    ):
        """Initialize LineSegmenter.

        Args:
            min_line_height (int): Minimum pixel height for a valid line crop.
            min_line_width (int): Minimum pixel width for a valid line crop.
            projection_threshold_ratio (float): Sensitivity factor for horizontal projection valleys.
        """
        self.min_line_height = min_line_height
        self.min_line_width = min_line_width
        self.projection_threshold_ratio = projection_threshold_ratio

    def _split_band_recursively(
        self,
        band_start: int,
        band_end: int,
        smoothed_proj: np.ndarray,
        min_h: int = 25,
        max_h: int = 120,
    ) -> List[Tuple[int, int]]:
        """Recursively split oversized multi-line bands only if truly multi-line."""
        h = band_end - band_start
        if h <= max_h:
            return [(band_start, band_end)]

        search_start = band_start + min_h
        search_end = band_end - min_h
        if search_start >= search_end:
            return [(band_start, band_end)]

        # Search for a distinct valley in the interior of the band
        sub_proj = smoothed_proj[search_start:search_end]
        min_val = np.min(sub_proj)
        max_val = np.max(smoothed_proj[band_start:band_end])

        # Only split if there is a real drop between lines (valley <= 40% of peak)
        if min_val > (max_val * 0.45):
            return [(band_start, band_end)]

        valley_idx = search_start + int(np.argmin(sub_proj))
        left = self._split_band_recursively(band_start, valley_idx, smoothed_proj, min_h, max_h)
        right = self._split_band_recursively(valley_idx, band_end, smoothed_proj, min_h, max_h)
        return left + right

    def segment_lines(
        self, binary_image: np.ndarray, source_image: np.ndarray
    ) -> List[LineSegment]:
        """Segment a binarized document page into individual horizontal text lines.

        Uses horizontal projection profiling with recursive valley splitting.

        Args:
            binary_image (np.ndarray): High contrast binarized image (255=background, 0=ink, or vice-versa).
            source_image (np.ndarray): Original or deskewed image from which to crop lines.

        Returns:
            List[LineSegment]: List of detected line segments ordered top-to-bottom.
        """
        h, w = binary_image.shape[:2]

        # Ensure inverted binary: ink pixels are 255, background is 0
        if np.mean(binary_image) > 127:
            inv = (255 - binary_image).astype(np.uint8)
        else:
            inv = binary_image.astype(np.uint8).copy()

        # Step 1: Detect and subtract horizontal ruled notebook lines
        hline_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
        hlines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, hline_kernel)
        cleaned_inv = cv2.subtract(inv, hlines)

        # Step 2: Remove small noise specks and huge non-text objects
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_inv)
        filtered_ink = np.zeros_like(cleaned_inv)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]

            # Discard tiny speck noise
            if area < 4:
                continue
            # Discard non-text massive objects (pens, large page borders)
            if area > (h * w * 0.08):
                continue
            # Discard large solid non-text objects
            if ch > 90 and cw > 250:
                continue

            filtered_ink[labels == i] = 255

        # Step 3: Morphological horizontal dilation to bridge characters into continuous lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        dilated = cv2.dilate(filtered_ink, kernel, iterations=1)

        # Horizontal projection profile: sum of ink pixels across each horizontal row
        proj = np.sum(dilated, axis=1) / 255.0
        smoothed = cv2.GaussianBlur(proj.reshape(-1, 1).astype(np.float32), (0, 0), sigmaX=0, sigmaY=2.0).flatten()

        max_val = np.max(smoothed) if np.max(smoothed) > 0 else 1.0
        threshold = max_val * self.projection_threshold_ratio

        # Identify contiguous horizontal bands where projection > threshold
        line_ranges: List[Tuple[int, int]] = []
        in_line = False
        start_y = 0

        for y in range(h):
            if smoothed[y] > threshold and not in_line:
                in_line = True
                start_y = y
            elif smoothed[y] <= threshold and in_line:
                in_line = False
                end_y = y
                if (end_y - start_y) >= self.min_line_height:
                    line_ranges.append((start_y, end_y))

        if in_line and (h - start_y) >= self.min_line_height:
            line_ranges.append((start_y, h))

        # Split any genuinely merged multi-line bands using median line height estimation
        raw_heights = [ey - sy for sy, ey in line_ranges if (ey - sy) >= self.min_line_height]
        median_lh = int(np.median(raw_heights)) if raw_heights else 35
        # Line height bounds: allow 0.6x to 1.7x median line height
        split_min_h = max(16, int(median_lh * 0.55))
        split_max_h = max(55, int(median_lh * 1.65))

        final_line_ranges: List[Tuple[int, int]] = []
        for sy, ey in line_ranges:
            sub_bands = self._split_band_recursively(
                sy, ey, smoothed, min_h=split_min_h, max_h=split_max_h
            )
            final_line_ranges.extend(sub_bands)

        # Fallback if projection detected nothing
        if not final_line_ranges:
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = [cv2.boundingRect(c) for c in contours]
            boxes = sorted([b for b in boxes if b[3] >= self.min_line_height and b[2] >= self.min_line_width], key=lambda b: b[1])
            segments: List[LineSegment] = []
            for idx, (bx, by, bw, bh) in enumerate(boxes):
                crop = source_image[by : by + bh, bx : bx + bw]
                segments.append(LineSegment(index=idx, x=bx, y=by, w=bw, h=bh, crop=crop))
            return segments

        # Build LineSegment instances for each horizontal slice
        segments: List[LineSegment] = []
        seg_idx = 0

        for sy, ey in final_line_ranges:
            lh = ey - sy
            if lh < self.min_line_height:
                continue

            # Compute horizontal bounds for the specific line
            line_slice_inv = filtered_ink[sy:ey, :]
            col_proj = np.sum(line_slice_inv, axis=0) / 255.0
            col_indices = np.where(col_proj > 0.5)[0]

            if len(col_indices) == 0:
                continue

            # Check character-like stroke transitions across median row
            mid_row = line_slice_inv[line_slice_inv.shape[0] // 2, :]
            transitions = np.sum(np.abs(np.diff(mid_row > 0))) // 2

            # Count character-sized connected components in this strip
            num_cc, _, stats_strip, _ = cv2.connectedComponentsWithStats(line_slice_inv)
            valid_cc = 0
            for ci in range(1, num_cc):
                c_h = stats_strip[ci, cv2.CC_STAT_HEIGHT]
                c_w = stats_strip[ci, cv2.CC_STAT_WIDTH]
                c_area = stats_strip[ci, cv2.CC_STAT_AREA]
                if c_h >= 7 and c_w >= 3 and c_area >= 12:
                    valid_cc += 1

            # Reject strips that lack text characteristics (solitary lines or smudge)
            if transitions < 4 and valid_cc < 3:
                continue

            sx = max(0, int(col_indices[0]) - 8)
            ex = min(w, int(col_indices[-1]) + 8)
            lw = ex - sx

            if lw >= self.min_line_width:
                # Add adaptive vertical padding to preserve ascenders and descenders (f, j, g, y, p, t)
                pad_y = max(6, int(lh * 0.16))
                py1 = max(0, sy - pad_y)
                py2 = min(h, ey + pad_y)
                crop = source_image[py1:py2, sx:ex]
                segments.append(
                    LineSegment(index=seg_idx, x=sx, y=py1, w=lw, h=(py2 - py1), crop=crop)
                )
                seg_idx += 1

        return segments

    def draw_segmentation_boxes(
        self,
        image: np.ndarray,
        segments: List[LineSegment],
        box_color: Tuple[int, int, int] = (0, 165, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        """Render colored bounding boxes over detected line segments for visual inspection.

        Args:
            image (np.ndarray): Base image to draw onto.
            segments (List[LineSegment]): Detected segments.
            box_color (Tuple[int, int, int]): BGR color for box outline.
            thickness (int): Line thickness in pixels.

        Returns:
            np.ndarray: Annotated RGB image copy.
        """
        # Ensure 3-channel RGB image for color overlay
        if image.ndim == 2:
            annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 1:
            annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            annotated = image.copy()

        for seg in segments:
            pt1 = (seg.x, seg.y)
            pt2 = (seg.x + seg.w, seg.y + seg.h)
            cv2.rectangle(annotated, pt1, pt2, box_color, thickness)
            label = f"#{seg.index + 1}"
            cv2.putText(
                annotated,
                label,
                (seg.x + 2, max(14, seg.y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 50, 50),
                1,
                cv2.LINE_AA,
            )

        return annotated

    def draw_word_boxes(
        self,
        image: np.ndarray,
        lines: list,
    ) -> np.ndarray:
        """Render individual green word bounding boxes with clean white text label tags.
        Matches the LearnOpenCV visual style requested by the user.

        Args:
            image (np.ndarray): Base image to draw onto.
            lines (list): List of LineRecognitionResult instances.

        Returns:
            np.ndarray: Annotated RGB image copy.
        """
        if image.ndim == 2:
            annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 1:
            annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            annotated = image.copy()

        h_img, w_img = annotated.shape[:2]

        for line in lines:
            words = getattr(line, "words", [])
            for w_res in words:
                wx, wy, ww, wh = getattr(w_res, "bbox", (0, 0, 0, 0))
                txt = getattr(w_res, "text", "")
                if ww < 4 or wh < 4 or not txt:
                    continue

                # 1. Emerald green word bounding box (RGB)
                box_color = (46, 204, 113)
                cv2.rectangle(annotated, (wx, wy), (wx + ww, wy + wh), box_color, 2)

                # 2. White badge pill with text directly above the word box
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.42
                thickness = 1
                (tw, th), _ = cv2.getTextSize(txt, font, font_scale, thickness)

                badge_y2 = max(th + 4, wy - 3)
                badge_y1 = max(0, badge_y2 - th - 5)
                badge_x1 = max(0, wx)
                badge_x2 = min(w_img - 1, wx + tw + 6)

                # Badge fill
                cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), (255, 255, 255), -1)
                # Badge border
                cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), (46, 204, 113), 1)
                # Word label text
                cv2.putText(
                    annotated,
                    txt,
                    (badge_x1 + 3, badge_y2 - 3),
                    font,
                    font_scale,
                    (15, 23, 42),
                    thickness,
                    cv2.LINE_AA,
                )

        return annotated
