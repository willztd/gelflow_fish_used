import time
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import argparse
from utils.models import *

def take_photo():

    # Set cameras
    cam0 = cv2.VideoCapture(0)  # left camera
    cam0.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cam0.set(3, 1280)
    cam0.set(4, 720)
    cam0.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    # change_x = -20
    # change_y = -10
    change_x = 10
    change_y = -20

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

def label_2_velocity(label):
    from heapq import nsmallest
    s = [0, np.pi]
    cos_course = label[0]
    sin_course = label[1]
    cos_course = np.clip(cos_course, -1, 1)
    sin_course = np.clip(sin_course, -1, 1)
    course = np.arccos(cos_course)
    if np.abs(sin_course) > 0.04:
        if sin_course < 0:
            course = 2 * np.pi - course
    else:
        course = np.array(nsmallest(1, s, key=lambda x: abs(x-course)))
    course = 180 * course / np.pi
    speed = label[2] * (500 - 150) + 150
    velocity = np.c_[course, speed]
    return velocity

def show_cam(cam):
    # Set cameras
    cam0 = cv2.VideoCapture(cam)  # left camera
    cam0.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cam0.set(3, 1280)
    cam0.set(4, 720)
    cam0.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    change_x = -5
    change_y = -7

    xmin = 590
    xmin += change_x
    xmax = xmin + 100
    ymin = 360
    ymin += change_y
    ymax = ymin + 100

    while True:
        time_start = time.time()
        # Read frames
        ret_val0, img = cam0.read()
        img_clip = img[ymin:ymax, xmin:xmax, :]
        img_gray = cv2.cvtColor(img_clip, cv2.COLOR_BGR2GRAY)
        motion_img = np.array(img_gray, dtype='uint8')
        cv2.imshow('image', img_gray)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cam0.release()
    cv2.destroyAllWindows()

def rotate(load_path):
    # import os
    # import cv2
    path = load_path
    filelist = os.listdir(path)  # 该文件夹下所有的文件（包括文件夹）
    for files in filelist:  # 遍历所有文件
        Olddir = os.path.join(path, files)  # 原来的文件路径
        if os.path.isdir(Olddir):  # 如果是文件夹则跳过
            continue
        img = cv2.imread(Olddir)  # 读取图片
        img = cv2.rotate(img, cv2.ROTATE_180)  # 旋转180度
        cv2.imwrite(Olddir, img)  # 保存图片

def rename(load_path):
    import os
    import re
    path = load_path
    filelist = os.listdir(path)  # 该文件夹下所有的文件（包括文件夹）
    for files in filelist:  # 遍历所有文件
        Olddir = os.path.join(path, files)  # 原来的文件路径
        if os.path.isdir(Olddir):  # 如果是文件夹则跳过
            continue
        filename = os.path.splitext(files)[0] # 文件名
        filetype = os.path.splitext(files)[1] # 文件扩展名
        newname = re.findall(r'\((.*?)\)', filename)[0]  # 提取括号内的内容
        Newdir = os.path.join(path, newname + filetype)  # 新的文件路径
        os.rename(Olddir, Newdir)  # 重命名

def get_flow(load_path):

    model = ConvLSTM(in_channel=1, conv_1=16, conv_2=32, fc_1=64, lstm_layers=1, lstm_hidden_size=64, cls_num=3)
    model.to(torch.device('cpu'))
    # load model
    model_path = load_path + '/model_best.pth.tar'
    assert os.path.isfile(model_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    seq_length = 5
    motion_seq = [np.zeros((100, 100),dtype=np.int64) for _x in range(seq_length)]

    time_num = 0
    course = 0
    speed = 0
    pred_c_cls = []
    pred_s_cls = []

    slides = []
    # image path
    im_dir = load_path + '\eval1'
    # get frames list
    filelist = os.listdir(im_dir)
    for image_dir in filelist:
        if os.path.splitext(image_dir)[1] == ".png" and os.path.basename(image_dir)[0] != '0':  # 后缀是tif, 前4个字符是abcc 的文件
            side = image_dir # os.path.join(im_dir, image_dir)
            slides.append(side)

    get_key = lambda i: int(i.split('.')[0])
    frames = sorted(slides, key=get_key)

    for frame in frames:
        f_path = os.path.join(im_dir, frame)
        image = cv2.imread(f_path)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        motion_img = np.array(img_gray, dtype='uint8')

        # switch to test mode
        model.eval()
        with torch.no_grad():
            # add img to motion_seq
            del (motion_seq[0])
            motion_seq.append(motion_img)

            motion = []
            for m in motion_seq:
                motion.append(torch.from_numpy(m).float())
            motion = torch.stack(motion)
            motion = torch.autograd.Variable(motion)
            motion = torch.unsqueeze(motion, dim=1)

            # model inference
            outputs = model(motion)

            # compute course, velocity
            pred_index = outputs.cpu().data.numpy()
            pred_index = pred_index[-1, :]
            pred_velocity = label_2_velocity(pred_index)
            pred_velocity = pred_velocity.reshape(-1)

        course = pred_velocity[0]
        speed = pred_velocity[1]
        pred_c_cls.append(course)
        pred_s_cls.append(speed)

        # course += pred_velocity[0]
        # speed += pred_velocity[1]
        # time_num += 1
        # if (time_num % 4) == 0:
        #     course = course / 4
        #     speed = speed / 4
        #     pred_c_cls.append(course)
        #     pred_s_cls.append(speed)
        #     course = 0
        #     speed = 0

        print(frame + " has been written!")

    test_x = np.arange(0, len(pred_s_cls))
    plt.plot(test_x, pred_c_cls, label='course')
    plt.savefig(load_path + '/' + str('0eval_course_error') + '.png')
    plt.show()

    with open('course_data_X4.txt', 'w') as f:
        f.write("pred_c_cls"+'\t'+'\t'+"pred_s_cls"+'\n')
        for i in range(len(pred_c_cls)):
            s = str(pred_c_cls[i])+'\t'+'\t'+str(pred_s_cls[i])+'\t'+'\n'
            f.write(s)
    f.close()

    print("done")

if __name__ == '__main__':
    # ## Options -------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--take_photo', dest='take_photo', action='store_true', help='take_photo')
    parser.add_argument('--plot_course', dest='plot_course', action='store_true', help='plot_course')
    parser.add_argument('--get_video', dest='get_video', action='store_true', help='get_video')
    parser.add_argument('--label_2_velocity', dest='label_2_velocity', action='store_true', help='label_2_velocity')
    parser.add_argument('--show_cam', dest='show_cam', action='store_true', help='show_cam')
    parser.add_argument('--eval_path', default=r'D:\will\Documents\ia.ac\04GelFlow\code\gelflow_fish_used\results\checkpoint_1010_tanh\eval_01_15_22_13', type=str, metavar= 'PATH', help='eval path)')

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
    if opt.label_2_velocity:
        course = 182.2
        course = course * np.pi / 180
        label_init = [np.cos(course), np.sin(course), 1]
        label = label_2_velocity(label_init)
        print(label)
    if opt.show_cam:
        show_cam(0)

    # rename(r'D:\will\Documents\ia.ac\04GelFlow\code\gelflow_fish_used\results\checkpoint_convlstm\eval1')
    # get_flow(r'D:\will\Documents\ia.ac\04GelFlow\code\gelflow_fish_used\results\checkpoint_convlstm')
    take_photo()
    # plot_course('./results/checkpoint_0901/eval')
    # plot_course(r'D:\will\Desktop\20230112\eval_01_12_15_29')
    # get_video(r'D:\will\Documents\ia.ac\04GelFlow\code\gelflow_fish_used\results\checkpoint_convlstm\eval1')
    # rename(r'D:\will\Documents\ia.ac\04GelFlow\code\gelflow_fish_used\results\checkpoint_convlstm\eval1')
    # rotate(r'D:\will\Documents\ia.ac\04GelFlow\code\gelflow_fish_used\results\checkpoint_convlstm\eval2')