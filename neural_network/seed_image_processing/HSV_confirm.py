import cv2
import numpy as np

image_path = "/media/ls/办公/ls/seeddata/seeddata/prosessed_imag/187.png"
img = cv2.imread(image_path)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

lower_purple = np.array([0, 0, 60])
upper_purple = np.array([255, 60, 255])
mask = cv2.inRange(hsv, lower_purple, upper_purple)
binary_gray = np.where(gray > 0, 255, 0).astype(np.uint8)

kernel = np.ones((3, 3), np.uint8)
kernel1 = np.ones((15, 15), np.uint8)
binary_gray = cv2.erode(binary_gray, kernel1, iterations=1)
mask_1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask_clean = cv2.morphologyEx(mask_1, cv2.MORPH_CLOSE, kernel1)
# kernel = np.ones((5, 5), np.uint8)
# mask_1 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# mask_clean = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, kernel)
mask_clean = cv2.bitwise_and(mask_clean, binary_gray)

result = cv2.bitwise_and(img, img, mask=mask_clean)

img = cv2.resize(img, (1000, 1000))
binary_gray = cv2.resize(binary_gray, (1000, 1000))
mask_closed = cv2.resize(mask_clean, (1000, 1000))
result = cv2.resize(result, (1000, 1000))

cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.namedWindow('Gray', cv2.WINDOW_NORMAL)
cv2.namedWindow('Clean Mask', cv2.WINDOW_NORMAL)
cv2.namedWindow('Purple Region', cv2.WINDOW_NORMAL)

cv2.imshow("Original", img)
cv2.imshow("Gray", binary_gray)
cv2.imshow("Clean Mask", mask_clean)
cv2.imshow("Purple Region", result)

cv2.waitKey(0)
cv2.destroyAllWindows()