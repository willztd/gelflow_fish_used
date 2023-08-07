import numpy as np
import torch
import torch.nn as nn
import os
import time
from utils.models import *
from utils.MarkersDetect import *
from utils.ImgDraw import *
import argparse
import random
import torch.backends.cudnn as cudnn
import cv2
from datetime import datetime
# import serial as ser
import struct
import re

def label_2_velocity(label):
    from heapq import nsmallest
    s = [0, np.pi]
    cos_course = label[:,0]
    sin_course = label[:,1]
    cos_course = np.clip(cos_course, -1, 1)
    sin_course = np.clip(sin_course, -1, 1)
    course = np.arccos(cos_course)
    for i in range(len(course)):
        if np.abs(sin_course[i]) > 0.04:
            if sin_course[i] < 0:
                course[i] = 2 * np.pi - course[i]
        else:
            course[i] = np.array(nsmallest(1, s, key=lambda x: abs(x-course[i])))
    course = 180 * course / np.pi
    speed = label[:, 2] * (500 - 150) + 150
    velocity = np.c_[course, speed]
    velocity = np.mean(velocity, axis=0)
    return velocity

def course_2_hex(course):
    input_s = 'f7 10 04 00 00 00 00 ff fd'
    input_s = input_s.strip()
    course = re.sub(r"(?<=\w)(?=(?:\w\w)+$)", " ", struct.pack('<f', course).hex())
    input_s = input_s[0:9] + course + input_s[20:]
    return input_s

if __name__ == '__main__':
    ## Options -------------------
    parser = argparse.ArgumentParser()

    parser.add_argument('--batchSize', default=16, type=int, metavar='N', help='input batch size')
    # GPU
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true', help='use cuda or not')

    parser.add_argument('--manualSeed', default=3963, type=int, help='manual seed')  # , default=4216

    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 4)')

    # Dataset
    parser.add_argument('--dataset', default='testset', type=str, help='which dataset to test, testset or trainset')

    # Test model
    parser.add_argument('--checkpoint', default='results/checkpoint_follow', type=str, metavar= \
        'PATH', help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--test_model', default='model_best.pth.tar', type=str, help='***.pth.tar')
    parser.add_argument('--model_arch', type=str, default='ConvLSTM', help='The model arch you selected')
    parser.add_argument('--seq_length', default=5, type=int, help='choose dataset path')
    parser.add_argument('--cam', default=0, type=int, help='choose camera path')
    parser.add_argument('--changex', default=-25, type=int, help='change_x')
    parser.add_argument('--changey', default=-10, type=int, help='change_y')

    opt = parser.parse_args()

    # Set cameras
    cam0 = cv2.VideoCapture(opt.cam)  # left camera
    cam0.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cam0.set(3, 1280)
    cam0.set(4, 720)
    cam0.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    change_x = opt.changex
    change_y = opt.changey

    xmin = 590
    xmin += change_x
    xmax = xmin + 100
    ymin = 360
    ymin += change_y
    ymax = ymin + 100

    Init_flag = False

    # Random seed
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)  #

    if opt.use_cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
        cudnn.benchmark = True
        cudnn.enabled = True

    args = vars(opt)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')


    ###---------set model arch and load par-----------------------
    # # Model
    if opt.model_arch == 'ConvLSTM':
        model = ConvLSTM(in_channel=1, conv_1=16, conv_2=32, fc_1=64, lstm_layers=1, lstm_hidden_size=64, cls_num=3)
    elif opt.model_arch == 'Conv':
        model = Conv(in_channel=6, conv_1=16, conv_2=32, cls_num=2)
    elif opt.model_arch == 'EasyLSTM':
        model = EasyLSTM(in_=20, lstm_layers=1, lstm_hidden_size=64, fc_1=32, cls_num=1)

    else:
        print('model selection error')
        exit()
    #
    # cuda
    if opt.use_cuda:
        model = model.cuda()
    else:
        model.to(torch.device('cpu'))

    # load model
    model_path = os.path.join(opt.checkpoint, opt.test_model)
    assert os.path.isfile(model_path), 'Error: no checkpoint directory found!'
    if opt.use_cuda:
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['state_dict'])

    print('Total params: %.4fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    ###----------------------------------------------------------

    seq_length = opt.seq_length
    motion_seq = [np.zeros((xmax-xmin, ymax-ymin),dtype=np.int64) for _x in range(seq_length)]

    pred_c_cls = []
    pred_s_cls = []
    now = datetime.now()
    timestr = now.strftime("%m_%d_%H_%M")
    eval_path = opt.checkpoint + '/eval_' + timestr
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    course_cv = np.zeros(5)
    course_nn = np.zeros(5)
    speed_nn = np.zeros(5)
    time_num = 0
    i = 0

    # Init Serial
    # se = ser.Serial("/dev/ttyTHS0", 115200)

    # Init BLobs detection
    print('Initializing...')

    while not Init_flag:
        ret_val0, img = cam0.read()

        blobs_center, img_draw, gray = blobs_detect_black_update(img.copy(), thresh=40, xmin=0, xmax=1280, ymin=0,
                                                               ymax=720)  # black

        # cv2.imshow('image', img_draw)
        # cv2.waitKey(1)

        if len(blobs_center) == 1:
            init_num = init_num + 1
        else:
            init_num = 0

        time.sleep(0.2)
        if init_num == 15:
            blobs_init = blobs_center
            Init_flag = True
            blobs_init_x = np.array(blobs_init)[0, 0]
            blobs_init_y = np.array(blobs_init)[0, 1]

            xmin = blobs_init_x - 50
            # xmin += change_x
            xmax = xmin + 100
            ymin = blobs_init_y - 50
            # ymin += change_y
            ymax = ymin + 100

            # img_clip = img[ymin:ymax, xmin:xmax, :]
            # blobs_init, _, _ = blobs_detect_black_update(img_clip.copy(), thresh=40, xmin=0, xmax=100, ymin=0, ymax=100)  # black

            print('Initializing done!')
            print(xmin, ymin)
            time.sleep(2)

    blobs_center_tmp = blobs_init

    # work
    time_start1 = time.time()

    while True:
        time_start = time.time()
        # Read frames
        ret_val0, img = cam0.read()
        img_clip = img[ymin:ymax, xmin:xmax, :]
        img_gray = cv2.cvtColor(img_clip, cv2.COLOR_BGR2GRAY)
        motion_img = np.array(img_gray, dtype='uint8')
        # cv2.imshow('image', img_gray)

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

            if opt.use_cuda:
                motion = motion.cuda()

            # model inference
            outputs = model(motion)

            # compute course, velocity
            pred_index = outputs.cpu().data.numpy()
            pred_velocity = label_2_velocity(pred_index)


        blobs_center, img_clip_, _ = blobs_detect_black_update(img_clip.copy(), thresh=40, xmin=0, xmax=100, ymin=0,
                                                           ymax=100, draw_flag=True)
        # show 2D motion
        if len(blobs_center) != 1:
            blobs_center = blobs_center_tmp
        diffL = 2 * np.array(blobs_center) - np.array([[50,50]])
        imgL = draw_arrow_2(img_clip_, blobs_center, diffL.tolist())
        cv2.imshow('2D motion', imgL)
        # key = cv2.waitKey(1)

        # get course
        diff = np.array(blobs_center) - np.array([[50,50]])
        delta_x = diff[0, 0]
        delta_y = diff[0, 1]

        if i < len(course_cv):
            course_cv[i] = get_course(delta_x, delta_y)

            course_nn[i] = pred_velocity[0]
            speed_nn[i] = pred_velocity[1]
            i = i + 1
        else:
            i = 0

        course_cv_mean = np.mean(np.array(course_cv), axis=0)
        course_nn_mean = np.mean(np.array(course_nn), axis=0)

        print(time.time()-time_start, ' cv:', course_cv_mean, ' nn:', course_nn_mean)

        blobs_center_tmp = blobs_center

        time_end = time.time()
        time_used = time_end - time_start
        if time_used < 0.05:
            time.sleep(0.05 - time_used)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # # init cam
    # ret_val0, img = cam0.read()
    #
    # time.sleep(5)
    #
    # # work
    # time_start1 = time.time()
    # while True:
    #     time_start = time.time()
    #     # Read frames
    #     ret_val0, img = cam0.read()
    #     img_clip = img[ymin:ymax, xmin:xmax, :]
    #     img_gray = cv2.cvtColor(img_clip, cv2.COLOR_BGR2GRAY)
    #     motion_img = np.array(img_gray, dtype='uint8')
    #     cv2.imshow('image', img_gray)
    #     cv2.waitKey(1)
    #
    #     # switch to test mode
    #     model.eval()
    #     with torch.no_grad():
    #         # add img to motion_seq
    #         del (motion_seq[0])
    #         motion_seq.append(motion_img)
    #
    #         motion = []
    #         for m in motion_seq:
    #             motion.append(torch.from_numpy(m).float())
    #         motion = torch.stack(motion)
    #         motion = torch.autograd.Variable(motion)
    #         motion = torch.unsqueeze(motion, dim=1)
    #
    #         if opt.use_cuda:
    #             motion = motion.cuda()
    #
    #         # model inference
    #         outputs = model(motion)
    #
    #         # compute course, velocity
    #         pred_index = outputs.cpu().data.numpy()
    #         pred_velocity = label_2_velocity(pred_index)
    #
    #     course += pred_velocity[0]
    #     speed += pred_velocity[1]
    #     time_num += 1
    #     cv2.imwrite(eval_path + '/' + str(time_num) + '.png', img_clip)
    #     time_end = time.time()
    #     time_used = time_end - time_start
    #     if time_used < 0.05:
    #         time.sleep(0.05 - time_used)
    #     # print(time_used, ':', pred_velocity)
    #
    #     if (time_num % 4) == 0:
    #         # cv2.imshow('image', img_gray)
    #         # cv2.waitKey(1)
    #         course = course / 4
    #         speed = speed / 4
    #         pred_c_cls.append(course)
    #         pred_s_cls.append(speed)
    #         print(time.time()-time_start1, ':', course)
    #         input_s = course_2_hex(course)
    #         send_list = []
    #         while input_s != '':
    #             num = int(input_s[0:2], 16)
    #             input_s = input_s[2:].strip()
    #             send_list.append(num)
    #         input_s = bytes(send_list)
    #         try:
    #             # se.write(input_s)
    #             # print(input_s)
    #             print('\n')
    #         except Exception:
    #             pass
    #         course = 0
    #         speed = 0
    #         time_start1 = time.time()
    #     if time_num > 6000:
    #         np.save(eval_path + '/course_error_' + str(i) + '.npy', pred_c_cls)
    #         np.save(eval_path + '/speed_error_' + str(i) + '.npy', pred_s_cls)
    #         print("save as ", '/course_error_' + str(i) )
    #         time_num = 0
    #         # i += 1
    #         break

    cam0.release()
    cv2.destroyAllWindows()









