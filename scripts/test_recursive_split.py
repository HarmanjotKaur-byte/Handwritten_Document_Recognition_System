import cv2
import numpy as np

def split_band_recursively(band_start, band_end, proj, min_h=20, max_h=38):
    h = band_end - band_start
    if h <= max_h:
        return [(band_start, band_end)]
    
    # We need to split because h > max_h
    # Search for the best valley in the middle region [band_start + min_h, band_end - min_h]
    search_start = band_start + min_h
    search_end = band_end - min_h
    if search_start >= search_end:
        # Cannot split further safely
        return [(band_start, band_end)]
    
    # Find local minima or minimum value in this range
    valley_idx = search_start + int(np.argmin(proj[search_start:search_end]))
    
    # Recursively split left and right halves
    left = split_band_recursively(band_start, valley_idx, proj, min_h, max_h)
    right = split_band_recursively(valley_idx, band_end, proj, min_h, max_h)
    return left + right

im = cv2.imread(r'C:/Users/hp/.gemini/antigravity/brain/cbe2ec0b-00ab-4ab3-bf33-aeae435587f8/.user_uploaded/media_1791632289142_7b8e8d94.png')
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
bg = cv2.medianBlur(cv2.dilate(gray, np.ones((5, 5), np.uint8)), 25)
diff = 255 - cv2.absdiff(gray, bg)
norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
_, bin_inv = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_inv)
ink_only = np.zeros_like(bin_inv)
for i in range(1, num_labels):
    area, cw, ch = stats[i, cv2.CC_STAT_AREA], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
    if 5 <= area <= (im.shape[0]*im.shape[1]*0.08) and not (cw>80 and ch<=3):
        ink_only[labels == i] = 255

dilated = cv2.dilate(ink_only, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)), iterations=1)
proj = np.sum(dilated > 0, axis=1)

# Smooth proj with Gaussian
smoothed = cv2.GaussianBlur(proj.reshape(-1, 1).astype(np.float32), (0, 0), sigmaX=0, sigmaY=2.0).flatten()

# Initial contiguous bands where smoothed > threshold
threshold = np.max(smoothed) * 0.15
bands = []
in_b = False
start_y = 0
for y in range(len(smoothed)):
    if smoothed[y] > threshold and not in_b:
        in_b = True
        start_y = y
    elif smoothed[y] <= threshold and in_b:
        in_b = False
        bands.append((start_y, y))
if in_b:
    bands.append((start_y, len(smoothed)))

print("Initial contiguous bands:", bands)
final_bands = []
for sy, ey in bands:
    sub = split_band_recursively(sy, ey, smoothed, min_h=22, max_h=45)
    final_bands.extend(sub)

print("Split final bands:")
for idx, (sy, ey) in enumerate(final_bands):
    print(f"  Band {idx}: y=[{sy}, {ey}], h={ey - sy}")
