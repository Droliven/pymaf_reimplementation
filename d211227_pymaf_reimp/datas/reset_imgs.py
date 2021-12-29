#!/usr/bin/env python
# encoding: utf-8
'''
@project : pymaf_reimp
@file    : reset_imgs.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-12-29 13:57
'''
import os
import os.path as osp
import shutil
import cv2
import numpy as np

def h36m():
    base_img_dir = r"G:\second_model_report_data\datas\three_dimension_reconstruction\h36m\images"
    out_imgs_dir = r"G:\second_model_report_data\datas\three_dimension_reconstruction\h36m\imgs"

    fs = os.listdir(base_img_dir)
    for img_path in fs:
        subject = img_path.split("_")[0]
        if not osp.exists(osp.join(out_imgs_dir, subject)):
            os.makedirs(osp.join(out_imgs_dir, subject))
        postfix = img_path[:-11]

        if not osp.exists(osp.join(out_imgs_dir, subject, postfix)):
            os.makedirs(osp.join(out_imgs_dir, subject, postfix))

        shutil.move(osp.join(base_img_dir, img_path), osp.join(out_imgs_dir, subject, postfix, f"{img_path}"))

    subjects = os.listdir(out_imgs_dir)
    for subj in subjects:
        cnt = 0
        actions = os.listdir(osp.join(out_imgs_dir, subj))
        for act in actions:
            cnt += len(os.listdir(osp.join(out_imgs_dir, subj, act)))
        print(f"{subj} {cnt}")

def read_calibration(calib_file, vid_list):
    Ks, Rs, Ts = [], [], []
    file = open(calib_file, 'r')
    content = file.readlines()
    for vid_i in vid_list:
        K = np.array([float(s) for s in content[vid_i * 7 + 5][11:-2].split()])
        K = np.reshape(K, (4, 4))
        RT = np.array([float(s) for s in content[vid_i * 7 + 6][11:-2].split()])
        RT = np.reshape(RT, (4, 4))
        R = RT[:3, :3]
        T = RT[:3, 3] / 1000
        Ks.append(K)
        Rs.append(R)
        Ts.append(T)
    return Ks, Rs, Ts


def mpi_inf_3dhp(subjects=[]):
    dataset_path = r"G:\second_model_report_data\datas\three_dimension_reconstruction\mpi_inf_3dhp\mpi_inf_3dhp_train_set"

    # training data
    user_list = subjects
    seq_list = range(1, 3)
    vid_list = list(range(3)) + list(range(4, 9))

    for user_i in user_list:
        for seq_i in seq_list:
            seq_path = os.path.join(dataset_path,
                                    'S' + str(user_i),
                                    'Seq' + str(seq_i))

            for j, vid_i in enumerate(vid_list):

                # image folder
                imgs_path = os.path.join(seq_path, 'imageFrames', 'video_' + str(vid_i))

                # extract frames from video file

                # if doesn't exist
                if not os.path.isdir(imgs_path):
                    os.makedirs(imgs_path)

                # video file
                vid_file = os.path.join(seq_path,
                                        'imageSequence',
                                        'video_' + str(vid_i) + '.avi')
                vidcap = cv2.VideoCapture(vid_file)

                # process video
                frame = 0
                while 1:
                    # extract all frames
                    success, image = vidcap.read()
                    if not success:
                        break
                    frame += 1
                    # image name
                    imgname = os.path.join(imgs_path, 'frame_%06d.jpg' % frame)
                    # save image
                    cv2.imwrite(imgname, image)
                print(f"S{user_i}-Seq{seq_i}-{frame}")



if __name__ == '__main__':
    mpi_inf_3dhp([1, 2])