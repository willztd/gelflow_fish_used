import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

def take_photo():

    # Set cameras
    cam0 = cv2.VideoCapture(0)  # left camera
    cam0.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cam0.set(3, 1280)
    cam0.set(4, 720)
    cam0.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    change_x = 0
    change_y = 0

    xmin = 590
    xmin += change_x
    xmax = xmin + 100
    ymin = 360
    ymin += change_y
    ymax = ymin + 100

    eval_flag = False

    ret_val0, img = cam0.read()
    img_clip = img[ymin:ymax, xmin:xmax, :]
    # cv2.imshow('image', img_clip)
    cv2.imwrite('./photo.png', img_clip)
    print("done")
    cam0.release()

def plot_course(load_path):
    course_path = load_path + '/course_error_'+str(0)+'.npy'
    speed_path = load_path + '/speed_error_' + str(0) + '.npy'
    pred_c_cls = np.load(course_path)
    pred_s_cls = np.load(speed_path)

    test_x = np.arange(0, len(pred_c_cls))
    plt.plot(test_x, pred_c_cls, label='course')
    plt.legend()
    plt.savefig(load_path + '/' + str('eval_course_error') + '.png')
    plt.clf()
    plt.plot(test_x, pred_s_cls, label='speed')
    plt.legend()
    plt.savefig(load_path + '/' + str('eval_speed_error') + '.png')


if __name__ == '__main__':
    # take_photo()
    plot_course('./results/checkpoint_0901/eval')
