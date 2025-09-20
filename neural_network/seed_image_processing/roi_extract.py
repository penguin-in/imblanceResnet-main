import cv2
import numpy as np
from scipy.spatial import ConvexHull

def detect_seed_head_tail(image_path, show=True):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad = cv2.convertScaleAbs(grad)

    _, binary1 = cv2.threshold(grad, 40, 255, cv2.THRESH_BINARY)  # 50 可调
    kernel = np.ones((5, 5), np.uint8)
    kernel1 = np.ones((5, 5), np.uint8)
    mask_1 = cv2.morphologyEx(binary1, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, kernel1)

    ys, xs = np.where(binary > 0)  # y 是行，x 是列
    points = np.stack([xs, ys], axis=1)  # shape (N, 2)

    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    max_dist = 0
    pt1 = pt2 = None
    n = len(hull_points)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(hull_points[i] - hull_points[j])
            if d > max_dist:
                max_dist = d
                pt1, pt2 = hull_points[i], hull_points[j]
    print("2")
    if pt1[1] > pt2[1]:
        pt1, pt2 = pt2, pt1
    bys, bxs = np.where(binary1 == 255)
    y_max_index = np.argmax(bys)
    y_min_index = np.argmin(bys)
    y_max = bys[y_max_index]
    x_max = bxs[y_max_index]
    y_min = bys[y_min_index]
    x_min = bxs[y_min_index]
    if pt1[1] > y_min + 50 or pt2[1] < y_max - 50:
        pt1 = [x_min, y_min]
        pt2 = [x_max, y_max]
    if show:

        color_img = img
        cv2.circle(color_img, tuple(pt1), 5, (0, 0, 255), -1)  # 红点
        cv2.circle(color_img, tuple(pt2), 5, (255, 0, 0), -1)  # 蓝点
        cv2.line(color_img, tuple(pt1), tuple(pt2), (0, 255, 0), 4)  # 绿线连接
        color_img = cv2.rectangle(color_img,(pt1[0]-112,pt1[1]),(pt1[0]+122,pt1[1]+224),(0, 0, 255), thickness=4)
        color_img = cv2.rectangle(color_img,(pt2[0]-112, pt2[1]-224),(pt2[0]+112,pt2[1]), (255, 0, 0), thickness=4)
        color_img = cv2.resize(color_img,(1000,1000))
        cv2.namedWindow('Max Distance Points', cv2.WINDOW_NORMAL)
        cv2.imshow("Max Distance Points", color_img)
        binary = cv2.resize(binary, (1000, 1000))
        cv2.namedWindow('binary', cv2.WINDOW_NORMAL)
        cv2.imshow("binary", binary)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 测试调用
if __name__ == "__main__":
    image_path = "/media/ls/办公/ls/seeddata/seeddata/prosessed_imag/1.png" # 修改为你的图片路径
    detect_seed_head_tail(image_path,show=True)