import time
from utils.ImgDraw import *
from utils.MarkersDetect import *

if __name__ == '__main__':
    # Set cameras
    cam0 = cv2.VideoCapture(0)  # left camera
    cam0.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cam0.set(3, 1280)
    cam0.set(4, 720)
    cam0.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    # Init BLobs detection
    print('Initializing...')
    Init_flag = False

    while not Init_flag:
        ret_val0, img = cam0.read()

        circles_center, img_draw_cir, _ = contours_detect_black_update(img.copy(), thresh=40, xmin=0, xmax=1280, ymin=0,
                                                                 ymax=720)  # black
        if len(circles_center) == 1:
            blobs_init_x = np.array(circles_center)[0, 0]
            blobs_init_y = np.array(circles_center)[0, 1]
            xmin = blobs_init_x - 50
            # xmin += change_x
            xmax = xmin + 100
            ymin = blobs_init_y - 50
            # ymin += change_y
            ymax = ymin + 100

        blobs_center, img_draw, gray = blobs_detect_black_update(img.copy(), thresh=40, xmin=0, xmax=1280, ymin=0,
                                                                 ymax=720)  # black
        if len(blobs_center) == 1:
            init_num = init_num + 1
        else:
            init_num = 0
        time.sleep(0.2)

        if init_num == 15:
            blobs_init = np.array(blobs_center) - np.array(circles_center) + np.array([[50, 50]])
            blobs_init = blobs_init.tolist()
            Init_flag = True

            time.sleep(2)
            print('Initializing done!')
            print('circles_center:', xmin, ymin)

    blobs_center_tmp = blobs_init
    course_cv = np.zeros(5)
    i = 0

    while True:
        time_start = time.time()
        # Read frames
        ret_val0, img = cam0.read()
        img_clip = img[ymin:ymax, xmin:xmax, :]
        img_gray = cv2.cvtColor(img_clip, cv2.COLOR_BGR2GRAY)
        motion_img = np.array(img_gray, dtype='uint8')
        # cv2.imshow('image', img_gray)

        blobs_center, img_clip_, _ = blobs_detect_black_update(img_clip.copy(), thresh=40, xmin=0, xmax=100, ymin=0,
                                                           ymax=100, draw_flag=True)
        # show 2D motion
        if len(blobs_center) != 1:
            blobs_center = blobs_center_tmp
        diffL = 2 * np.array(blobs_center) - np.array(blobs_init)
        imgL = draw_arrow_2(img_clip_, blobs_center, diffL.tolist())
        cv2.imshow('2D motion', imgL)

        # get course
        diff = np.array(blobs_center) - np.array(blobs_init)
        delta_x = diff[0, 0]
        delta_y = diff[0, 1]

        if i < len(course_cv):
            course_cv[i] = get_course(delta_x, delta_y)
            i = i + 1
        else:
            i = 0

        course_cv_mean = np.mean(np.array(course_cv), axis=0)

        print(time.time()-time_start, ' cv:', course_cv_mean)

        blobs_center_tmp = blobs_center

        time_end = time.time()
        time_used = time_end - time_start
        if time_used < 0.05:
            time.sleep(0.05 - time_used)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
