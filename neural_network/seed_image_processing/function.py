import cv2
import numpy as np
from scipy.spatial import ConvexHull
from tensorboard.compat.tensorflow_stub.dtypes import uint8

from neural_network.seed_image_processing.seed_belly_extract import holes_filled, eroded, labels


def extract_color_region(
    image_path,
    lower_hsv=(0, 0, 70),
    upper_hsv=(20, 255, 255),
    lower_hsv_destroy=(0, 0, 87),
    upper_hsv_destroy=(255, 60, 255),
    resize_shape=(1000, 1000),
    kernel_size=(5, 5),
    bw_threshold = 40,
    show=False
):
    img = cv2.imread(image_path)
    original_img = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw_image = np.where(gray > 0, 255, 0).astype(np.uint8)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad = cv2.convertScaleAbs(grad)

    _, binary1 = cv2.threshold(grad, bw_threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    kernel1 = np.ones((5, 5), np.uint8)
    mask_1 = cv2.morphologyEx(binary1, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, kernel1)
    binary2 = binary.copy()
    binary2[640 - 200:640 + 200, :] = 0
    ys, xs = np.where(binary2 > 0)
    points = np.stack([xs, ys], axis=1)
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


    head_image = img[pt1[1]:pt1[1] + 224, pt1[0] - 112:pt1[0] + 122, :].copy()
    tail_image = img[pt2[1] - 224:pt2[1], pt2[0] - 112:pt2[0] + 112, :].copy()

    img[pt1[1]:pt1[1] + 224, pt1[0] - 112:pt1[0] + 122, :] = 0
    img1 = img.copy()
    img[pt2[1] - 224:pt2[1], pt2[0] - 112:pt2[0] + 112, :] = 0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
    kernel = np.ones(kernel_size, np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_CLOSE, kernel)
    hsv_destroy = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    gray_destroy = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    binary_destroy = np.where(gray_destroy > 0, 255, 0).astype(np.uint8)
    kernel_destroy = np.ones((15,15), np.uint8)
    binary_destroy = cv2.erode(binary_destroy, kernel_destroy, iterations=1)
    mask_destroy = cv2.inRange(hsv_destroy,lower_hsv_destroy, upper_hsv_destroy)
    mask_open_destroy = cv2.morphologyEx(mask_destroy, cv2.MORPH_OPEN, kernel1)
    mask_closed_destroy = cv2.morphologyEx(mask_open_destroy, cv2.MORPH_CLOSE, kernel_destroy)
    result_destroy = cv2.bitwise_and(mask_closed_destroy, binary_destroy)
    if show:
        color_img = original_img.copy()
        cv2.circle(color_img, tuple(pt1), 5, (0, 0, 255), -1)  # 红点
        cv2.circle(color_img, tuple(pt2), 5, (255, 0, 0), -1)  # 蓝点
        cv2.line(color_img, tuple(pt1), tuple(pt2), (0, 255, 0), 2)  # 绿线连接
        print("3")
        color_img = cv2.rectangle(color_img, (pt1[0] - 112, pt1[1]), (pt1[0] + 122, pt1[1] + 224), (0, 0, 255))
        color_img = cv2.rectangle(color_img, (pt2[0] - 112, pt2[1] - 224), (pt2[0] + 112, pt2[1]), (255, 0, 0))

        original_img = cv2.resize(original_img, (1280, 1280))
        cv2.namedWindow('original_img', cv2.WINDOW_NORMAL)
        cv2.imshow("original_img", original_img)

        color_img = cv2.resize(color_img, (1280, 1280))
        cv2.namedWindow('color_img', cv2.WINDOW_NORMAL)
        cv2.imshow("color_img", color_img)

        head_image = cv2.resize(head_image, (224, 224))
        cv2.namedWindow('head_image', cv2.WINDOW_NORMAL)
        cv2.imshow("head_image", head_image)

        tail_image = cv2.resize(tail_image, (224, 224))
        cv2.namedWindow('tail_image', cv2.WINDOW_NORMAL)
        cv2.imshow("tail_image", tail_image)

        mask_clean = cv2.resize(mask_clean, (224, 224))
        cv2.namedWindow('mask_clean', cv2.WINDOW_NORMAL)
        cv2.imshow("mask_clean", mask_clean)

        result_destroy = cv2.resize(result_destroy, (224, 224))
        cv2.namedWindow('result_destroy', cv2.WINDOW_NORMAL)
        cv2.imshow("result_destroy", result_destroy)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return mask_clean,head_image,tail_image,result_destroy,bw_image

def accuracy(pred, true):
    pred = np.array(pred).flatten()
    true = np.array(true).flatten()

    return np.mean(pred == true)

def seed_belly_extract(image_path,show = False):
    img = cv2.imread(image_path)

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    s_eq = cv2.equalizeHist(s)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge((h,s_eq,v_eq))
    # img_eq = cv2.cvtColor(hsv_eq,cv2.COLOR_HSV2RGB)

    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    lower_belly = np.array([0,0,140])
    upper_belly = np.array([255,140,255])
    mask = cv2.inRange(hsv_eq,lower_belly,upper_belly)

    hole_filled = mask.copy()
    h_image,w_image = hole_filled.shape
    mask_floodfill = np.zeros((h_image+2,w_image+2),np.uint8)
    cv2.floodFill(hole_filled,mask_floodfill,(0,0),255)
    hole_filled_inv = cv2.bitwise_not(hole_filled)
    mask = cv2.bitwise_or(mask,hole_filled_inv)

    binary_gray = np.where(gray > 0, 255, 0).astype(np.uint8)
    kernel_3 = np.ones((3,3),np.uint8)
    kernel_15 = np.ones((15,15),np.uint8)
    binary_gray = cv2.erode(binary_gray,kernel_15,iterations=1)

    mask_clean = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel_3)
    mask_clean = cv2.morphologyEx(mask_clean,cv2.MORPH_OPEN,kernel_15)
    mask_clean = cv2.bitwise_and(mask_clean,binary_gray)

    kernel_50 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(50,50))
    kernel_60 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(60,60))
    eroded = cv2.erode(mask_clean,kernel_50,iterations=1)
    num_labels,labels,stats,_ = cv2.connectedComponentsWithStats(eroded,connectivity=8)
    max_label = 1
    max_area = 0
    for i in range(1,num_labels):
        area = stats[i,cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            max_label = i
    eroded_mask = np.uint8(labels == max_label)*255
    dilated = cv2.dilate(eroded_mask,kernel_60,iterations=1)
    dilated = cv2.morphologyEx(dilated,cv2.MORPH_CLOSE,kernel_50)

    holes_filled = dilated.copy()
    h, w = holes_filled.shape
    mask_floodfill = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(holes_filled, mask_floodfill, (0, 0), 255)

    holes_filled_inv = cv2.bitwise_not(holes_filled)

    dilated = cv2.bitwise_or(dilated, holes_filled_inv)

    result_image = cv2.bitwise_and(img,img,mask=dilated)

    if show:
        img_show = cv2.resize(img, (1000, 1000))
        dilated_show = cv2.resize(dilated, (1000, 1000))
        mask_show = cv2.resize(mask, (1000, 1000))
        result_show = cv2.resize(result_image, (1000, 1000))

        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        cv2.namedWindow('dilated', cv2.WINDOW_NORMAL)
        cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
        cv2.namedWindow('result', cv2.WINDOW_NORMAL)

        cv2.imshow("Original", img_show)
        cv2.imshow("dilated", dilated_show)
        cv2.imshow("mask", mask_show)
        cv2.imshow("result", result_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result_image

def aspect_ratio_min_area(bin_img):

    img = (bin_img > 0).astype('uint8') * 255
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    all_pts = np.vstack(contours)
    rect = cv2.minAreaRect(all_pts)
    (cx, cy), (w, h), angle = rect
    if w == 0 or h == 0:
        return None
    long_side = max(w, h)
    short_side = min(w, h)
    return long_side / short_side














