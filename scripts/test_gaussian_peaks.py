import cv2
import numpy as np

im = cv2.imread(r'C:/Users/hp/.gemini/antigravity/brain/cbe2ec0b-00ab-4ab3-bf33-aeae435587f8/.user_uploaded/media_1791632289142_7b8e8d94.png')
h, w = im.shape[:2]
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

bg = cv2.dilate(gray, np.ones((5, 5), np.uint8))
bg = cv2.medianBlur(bg, 25)
diff = 255 - cv2.absdiff(gray, bg)
norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
_, bin_inv = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_inv)
ink_only = np.zeros_like(bin_inv)
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    cw = stats[i, cv2.CC_STAT_WIDTH]
    ch = stats[i, cv2.CC_STAT_HEIGHT]
    if area < 5 or area > (h * w * 0.08): continue
    if cw > 80 and ch <= 3: continue
    if ch > 60 and cw > 200: continue
    ink_only[labels == i] = 255

proj_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
dilated_ink = cv2.dilate(ink_only, proj_kernel, iterations=1)
row_proj = np.sum(dilated_ink > 0, axis=1).astype(np.float32)

# Gaussian smoothing with sigma = 4.0
smoothed = cv2.GaussianBlur(row_proj.reshape(-1, 1), (0, 0), sigmaX=0, sigmaY=4.0).flatten()

# Find major peaks (peaks that are at least 40 and higher than neighbors by distance >= 20)
from scipy.signal import find_peaks

peaks, peak_props = find_peaks(smoothed, height=40, distance=20)
valleys, _ = find_peaks(-smoothed, distance=20)

print("Gaussian Smoothed Peaks at y:", peaks)
for p in peaks:
    print(f"  Peak at y={p}: height={smoothed[p]:.1f}")

# Find valleys between consecutive peaks
split_points = []
for i in range(len(peaks) - 1):
    p1 = peaks[i]
    p2 = peaks[i+1]
    v_idx = p1 + np.argmin(smoothed[p1:p2])
    split_points.append(v_idx)
    print(f"  Valley between y={p1} and y={p2} is at y={v_idx} (val={smoothed[v_idx]:.1f})")

print("All split points:", split_points)
