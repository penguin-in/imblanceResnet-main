import cv2
import numpy as np

image_path = "/media/ls/办公/ls/seeddata/seeddata/prosessed_imag/92.png"
img = cv2.imread(image_path)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv)
h_eq = cv2.equalizeHist(h)
v_eq = cv2.equalizeHist(v)
s_eq = cv2.equalizeHist(s)
hsv_eq = cv2.merge((h_eq,s_eq,v_eq))
# img_eq = cv2.cvtColor(hsv_eq,cv2.COLOR_HSV2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

lower_purple = np.array([0, 0, 130])
upper_purple = np.array([255, 150, 255])
kernel_15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
mask = cv2.inRange(hsv_eq, lower_purple, upper_purple)
mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel_15)
holes_filled = mask.copy()

h, w = holes_filled.shape
mask_floodfill = np.zeros((h + 2, w + 2), np.uint8)
cv2.floodFill(holes_filled, mask_floodfill, (0, 0), 255)

holes_filled_inv = cv2.bitwise_not(holes_filled)

mask = cv2.bitwise_or(mask, holes_filled_inv)

binary_gray = np.where(gray > 0, 255, 0).astype(np.uint8)
kernel = np.ones((3, 3), np.uint8)
kernel1 = np.ones((15, 15), np.uint8)
binary_gray = cv2.erode(binary_gray, kernel1, iterations=1)

mask_1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask_clean = cv2.morphologyEx(mask_1, cv2.MORPH_CLOSE, kernel1)
mask_clean = cv2.bitwise_and(mask_clean, binary_gray)

kernel_25 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
kernel_60 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))
eroded = cv2.erode(mask_clean, kernel_25, iterations=1)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)
max_area = 0
max_label = 1
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if area > max_area:
        max_area = area
        max_label = i
eroded_mask = np.uint8(labels == max_label) * 255
dilated = cv2.dilate(eroded_mask, kernel_60, iterations=1)
dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_25)

holes_filled = dilated.copy()
h, w = holes_filled.shape
mask_floodfill = np.zeros((h + 2, w + 2), np.uint8)
cv2.floodFill(holes_filled, mask_floodfill, (0, 0), 255)

holes_filled_inv = cv2.bitwise_not(holes_filled)

dilated = cv2.bitwise_or(dilated, holes_filled_inv)

result = cv2.bitwise_and(img, img, mask=dilated)

img = cv2.resize(img, (1000, 1000))
dilated= cv2.resize(dilated, (1000, 1000))
dilated_filled = cv2.resize(mask,(1000, 1000))

cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.namedWindow('dilated', cv2.WINDOW_NORMAL)
cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
cv2.namedWindow('result', cv2.WINDOW_NORMAL)

cv2.imshow("Original", img)
cv2.imshow("dilated", dilated)
cv2.imshow("mask", mask)
cv2.imshow("result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()