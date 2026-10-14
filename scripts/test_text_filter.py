import cv2
import numpy as np

im = cv2.imread(r'C:/Users/hp/.gemini/antigravity/brain/cbe2ec0b-00ab-4ab3-bf33-aeae435587f8/.user_uploaded/media_1791632289142_7b8e8d94.png')
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
bg = cv2.medianBlur(cv2.dilate(gray, np.ones((5, 5), np.uint8)), 25)
diff = 255 - cv2.absdiff(gray, bg)
norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
_, bin_inv = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

test_bands = [
    (86, 120, "Line 1 (Text)"),
    (120, 155, "Line 2 (Text)"),
    (155, 185, "Line 3 (Text)"),
    (200, 246, "Pen / Icons (Non-text)"),
    (61, 69, "Ruled line remnant"),
]

for sy, ey, name in test_bands:
    strip = bin_inv[sy:ey, :]
    # Count horizontal ink transitions on median row
    mid_row = strip[strip.shape[0] // 2, :]
    transitions = np.sum(np.abs(np.diff(mid_row > 0))) // 2
    
    # Also count connected components inside this strip
    num_cc, _, stats, _ = cv2.connectedComponentsWithStats(strip)
    # Filter valid character-sized components
    valid_cc = 0
    for i in range(1, num_cc):
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        area = stats[i, cv2.CC_STAT_AREA]
        if ch >= 8 and cw >= 3 and area >= 15:
            valid_cc += 1
            
    print(f"{name}: transitions={transitions}, valid_cc={valid_cc}")
