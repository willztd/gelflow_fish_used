import cv2
import numpy as np

def blobs_detect_black_update(img, thresh=120, xmin=0, xmax=1280, ymin=0, ymax=720, draw_flag=False):
    # cv2.BORDER_CONSTANT = 180

    img_clip = img[ymin:ymax, xmin:xmax, :]

    gray = cv2.cvtColor(img_clip, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)

    raw_image_blur = gray
    # cv2.imshow('img blur', raw_image_blur.astype(np.uint8))

    # ref_blur = cv2.GaussianBlur(gray.astype(np.float32), (51, 51), 30)
    # # cv2.imshow('ref blur', ref_blur.astype(np.uint8))

    img_diff = raw_image_blur
    diff_min = 50
    diff_max = 170

    img_diff = (img_diff - diff_min) / (diff_max - diff_min)
    img_diff = np.clip(img_diff, 0, 1) * 255
    img_diff = img_diff.astype(np.uint8)
    # cv2.imshow('img diff', img_diff)


    ret, img_diff = cv2.threshold(img_diff, thresh, 255, cv2.THRESH_BINARY)  # 88

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img_diff = cv2.erode(img_diff, kernel, iterations=1)
    img_diff = cv2.dilate(img_diff, kernel, iterations=1)

    # cv2.imshow('img bi', img_diff)

    params = cv2.SimpleBlobDetector_Params()

    params.filterByCircularity = True
    params.minCircularity = 0.4
    params.filterByConvexity = True
    params.minConvexity = 0.7

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img_diff)

    blob_center = []
    for i in range(len(keypoints)):
        x = int(keypoints[i].pt[0]) + xmin
        y = int(keypoints[i].pt[1]) + ymin
        blob_center.append([x, y])

    if draw_flag:   # 是否绘制blobs
        point_size = 2
        point_color = (0, 0, 255)  # BGR
        thickness = 2  # 180 、4、8
        for ce in blob_center:
            cv2.circle(img, (int(ce[0]), int(ce[1])), point_size, point_color, thickness)
        # print(len(blob_center))
    return blob_center, img, img_diff

# cv2.HoughCircles(image, method, dp, minDist, circles, param1, param2, minRadius, maxRadius)
# https://zhuanlan.zhihu.com/p/333968828
# https://blog.csdn.net/weixin_42272768/article/details/125218619
def contours_detect_black_update(img, thresh=120, xmin=0, xmax=1280, ymin=0, ymax=720, draw_flag=False):
    # cv2.BORDER_CONSTANT = 180

    img_clip = img[ymin:ymax, xmin:xmax, :]

    gray = cv2.cvtColor(img_clip, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)

    raw_image_blur = gray
    # cv2.imshow('img blur', raw_image_blur.astype(np.uint8))

    # ref_blur = cv2.GaussianBlur(gray.astype(np.float32), (51, 51), 30)
    # # cv2.imshow('ref blur', ref_blur.astype(np.uint8))

    img_diff = raw_image_blur
    diff_min = 50
    diff_max = 170

    img_diff = (img_diff - diff_min) / (diff_max - diff_min)
    img_diff = np.clip(img_diff, 0, 1) * 255
    img_diff = img_diff.astype(np.uint8)
    # cv2.imshow('img diff', img_diff)
    # cv2.waitKey(1)

    # th, img_diff = cv2.threshold(img_diff, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_TRUNC + cv2.THRESH_OTSU)

    circles = cv2.HoughCircles(image=img_diff, method=cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=50, param2=50, minRadius=15, maxRadius=50)

    circles_center = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for i in range(len(circles)):
            x = int(circles[i][0]) + xmin
            y = int(circles[i][1]) + ymin
            circles_center.append([x, y])

        if draw_flag:   # 是否绘制blobs
            for (x, y, r) in circles:
                cv2.circle(img, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    return circles_center, img, img_diff