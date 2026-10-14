import cv2
import numpy as np

im = cv2.imread(r'C:/Users/hp/.gemini/antigravity/brain/cbe2ec0b-00ab-4ab3-bf33-aeae435587f8/.user_uploaded/media_1791632289142_7b8e8d94.png')
h, w = im.shape[:2]
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# Check illumination and adaptive thresholding
# In notebook photos, paper brightness can vary across the page.
# Using background division + Otsu gives clean ink:
bg = cv2.dilate(gray, np.ones((5, 5), np.uint8))
bg = cv2.medianBlur(bg, 25)
diff = 255 - cv2.absdiff(gray, bg)
norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Threshold to get ink mask (255 = ink, 0 = paper)
_, bin_inv = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Connected components analysis to remove non-text:
# 1. Very large solid objects (like pens, dark blocks)
# 2. Ruled notebook lines (very wide, but height <= 3 px)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_inv)
ink_only = np.zeros_like(bin_inv)

for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    cw = stats[i, cv2.CC_STAT_WIDTH]
    ch = stats[i, cv2.CC_STAT_HEIGHT]
    top = stats[i, cv2.CC_STAT_TOP]
    left = stats[i, cv2.CC_STAT_LEFT]
    
    # Ignore tiny salt-and-pepper noise
    if area < 5:
        continue
    # Ignore solid non-text objects (pen barrel: huge area or very high density block)
    if area > (h * w * 0.08):
        continue
    # Ruled lines: very wide (cw > 100) and very thin (ch <= 3)
    if cw > 80 and ch <= 3:
        continue
    # Pen body: check if it's in the lower portion and has very large bounding box
    if ch > 60 and cw > 200:
        continue
        
    ink_only[labels == i] = 255

# Now let's inspect horizontal projection of ink_only
# We want to separate text lines even when they are close
# Use horizontal projection with peak/valley analysis:
# Dilate slightly horizontally to bridge characters within words, but NOT vertically
proj_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
dilated_ink = cv2.dilate(ink_only, proj_kernel, iterations=1)

row_proj = np.sum(dilated_ink > 0, axis=1)

# Smooth row_proj
kernel_size = 7
kernel = np.ones(kernel_size) / kernel_size
smoothed_proj = np.convolve(row_proj, kernel, mode='same')

print("Smoothed proj max:", np.max(smoothed_proj))
for y in range(0, h, 10):
    bar = '*' * int(smoothed_proj[y] / 10)
    print(f"y={y:3d}: {smoothed_proj[y]:5.1f} {bar}")
