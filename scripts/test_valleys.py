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

# Connected components filtering
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_inv)
ink_only = np.zeros_like(bin_inv)
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    cw = stats[i, cv2.CC_STAT_WIDTH]
    ch = stats[i, cv2.CC_STAT_HEIGHT]
    if area < 5 or area > (h * w * 0.08): continue
    if cw > 80 and ch <= 3: continue  # thin ruled lines
    if ch > 60 and cw > 200: continue # pen body
    ink_only[labels == i] = 255

proj_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
dilated_ink = cv2.dilate(ink_only, proj_kernel, iterations=1)
row_proj = np.sum(dilated_ink > 0, axis=1)

# Smooth
smoothed = np.convolve(row_proj, np.ones(7)/7.0, mode='same')

# Find local maxima (peaks) and minima (valleys)
peaks = []
valleys = []
for y in range(2, h - 2):
    if smoothed[y] > smoothed[y-1] and smoothed[y] >= smoothed[y+1] and smoothed[y] > 30:
        peaks.append((y, smoothed[y]))
    elif smoothed[y] < smoothed[y-1] and smoothed[y] <= smoothed[y+1]:
        valleys.append((y, smoothed[y]))

print("Detected peaks (y, val):", peaks)
print("Detected valleys (y, val):", valleys)

# For each adjacent pair of peaks, find the minimum between them
splits = []
for i in range(len(peaks) - 1):
    y1 = peaks[i][0]
    y2 = peaks[i+1][0]
    if y2 - y1 >= 18:  # at least minimum line separation
        # Find minimum in [y1, y2]
        min_y = y1 + int(np.argmin(smoothed[y1:y2]))
        splits.append(min_y)

print("Split y points between lines:", splits)
