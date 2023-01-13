import cv2
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse

def take_photo():

    # Set cameras
    cam0 = cv2.VideoCapture(0)  # left camera
    cam0.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cam0.set(3, 1280)
    cam0.set(4, 720)
    cam0.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    change_x = -20
    change_y = -10

    xmin = 590
    xmin += change_x
    xmax = xmin + 100
    ymin = 360
    ymin += change_y
    ymax = ymin + 100

    time.sleep(5)

    i = 0
    while i < 5:
        ret_val0, img = cam0.read()
        img_clip = img[ymin:ymax, xmin:xmax, :]
        # cv2.imshow('image', img_clip)
        cv2.imwrite('./photo' + str(i) + '.png', img_clip)
        print(i)
        i += 1
        time.sleep(1)
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
    plt.savefig(load_path + '/' + str('0eval_course_error') + '.png')
    plt.clf()
    plt.plot(test_x, pred_s_cls, label='speed')
    plt.legend()
    plt.savefig(load_path + '/' + str('0eval_speed_error') + '.png')

def get_video(load_path):
    slides = []
    # image path
    im_dir = load_path
    # output video path
    save_video_dir = load_path
    if not os.path.exists(save_video_dir):
        os.makedirs(save_video_dir)
    # set saved fps
    fps = 20
    # get frames list
    filelist = os.listdir(im_dir)
    for image_dir in filelist:
        if os.path.splitext(image_dir)[1] == ".png" and os.path.basename(image_dir)[0] != '0':  # 后缀是tif, 前4个字符是abcc 的文件
            side = image_dir # os.path.join(im_dir, image_dir)
            slides.append(side)

    get_key = lambda i: int(i.split('.')[0])
    frames = sorted(slides, key=get_key)
    # w,h of image
    img = cv2.imread(os.path.join(im_dir, frames[0]))
    img_size = (img.shape[1], img.shape[0])
    # get seq name
    seq_name = os.path.basename(load_path)
    # splice video_dir
    video_dir = save_video_dir + '/0' + seq_name + '.avi'
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # also can write like:fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # if want to write .mp4 file, use 'MP4V'
    videowriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

    for frame in frames:
        f_path = os.path.join(im_dir, frame)
        image = cv2.imread(f_path)
        videowriter.write(image)
        print(frame + " has been written!")

    print("videos saved at:", video_dir)
    videowriter.release()

if __name__ == '__main__':
    ## Options -------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--take_photo', dest='take_photo', action='store_true', help='take_photo')
    parser.add_argument('--plot_course', dest='plot_course', action='store_true', help='plot_course')
    parser.add_argument('--get_video', dest='get_video', action='store_true', help='get_video')
    parser.add_argument('--eval_path', default=r'D:\will\Desktop\20230112\eval_01_12_21_16', type=str, metavar= 'PATH', help='eval path)')
    opt = parser.parse_args()

    args = vars(opt)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    if opt.take_photo:
        take_photo()
    if opt.plot_course:
        plot_course(opt.eval_path)
    if opt.get_video:
        get_video(opt.eval_path)

    # take_photo()
    # plot_course('./results/checkpoint_0901/eval')
    # plot_course(r'D:\will\Desktop\20230112\eval_01_12_15_29')
    # get_video(r'D:\will\Desktop\20230112\eval_01_12_21_16')

