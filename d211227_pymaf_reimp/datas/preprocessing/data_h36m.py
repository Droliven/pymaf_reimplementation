#!/usr/bin/env python
# encoding: utf-8
'''
@project : pymaf_reimp
@file    : data_h36m.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-12-30 20:50
'''

import os.path as osp
import numpy as np
import os
import shutil
import cv2

def reset_dirs_h36m():
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

def change_h36m_eval_npz():
    # npz_path = osp.join(r"G:\second_model_report_data\datas\three_dimension_reconstruction\spin_pymaf_data\dataset_extras\h36m_valid_protocol1.npz")
    npz_path = osp.join(r"G:\second_model_report_data\datas\three_dimension_reconstruction\spin_pymaf_data\dataset_extras\h36m_valid_protocol2.npz")
    # ds = "h36m_p1"
    ds = "h36m_p2"
    outputs_path = r"H:\h36m"

    # 顺序遍历检索
    npz_data = np.load(npz_path, allow_pickle=True)
    print(f"Train >>> {ds}: {npz_data.files}, shape: {npz_data['imgname'].shape[0]}",
          {npz_data['imgname'][0]})  # ['imgname', 'center', 'scale', 'S']

    new_imgname = []
    for i in range(npz_data['imgname'].shape[0]):
        old_v = npz_data['imgname'][i]  # images/S9_Directions_1.54138969_000001.jpg
        frame_idx = old_v.split("/")[1]
        subj = frame_idx.split("_")[0]
        act = frame_idx[:-11]
        new_v = subj + "/" + act + "/" + frame_idx
        new_imgname.append(new_v)

    # np.savez(os.path.join(outputs_path, "h36m_valid_protocol1.npz"), imgname=np.array(new_imgname), center=npz_data["center"], scale=npz_data["scale"], S=npz_data["S"])
    np.savez(os.path.join(outputs_path, "h36m_valid_protocol2.npz"), imgname=np.array(new_imgname), center=npz_data["center"], scale=npz_data["scale"], S=npz_data["S"])

def del_notused_h36mp2mosh_test():
    indir = r"G:\second_model_report_data\datas\three_dimension_reconstruction\h36m\imgs"
    npzpath = r"G:\second_model_report_data\datas\three_dimension_reconstruction\spin_pymaf_data\dataset_extras\h36m_mosh_valid_p2.npz"
    outpath = osp.join(r"H:\h36m\imgs")
    if not osp.exists(outpath):
        os.makedirs(outpath)

    imgname = np.load(npzpath, allow_pickle=True)["imgname"]
    print(imgname.shape[0])

    img_cnt = 0
    for i in range(imgname.shape[0]):
        dirs = imgname[i].split("/") # S9/S9_Directions_1.60457274/S9_Directions_1.60457274_000001.jpg
        subj = dirs[0]
        act = dirs[1]
        imageidx = dirs[-1]

        if not osp.exists(osp.join(outpath, subj, act)):
            os.makedirs(osp.join(outpath, subj, act))

        if not osp.exists(osp.join(outpath, subj, act, imageidx)) and osp.exists(osp.join(indir, subj, act, imageidx)):
            shutil.copyfile(osp.join(indir, subj, act, imageidx), osp.join(outpath, subj, act, imageidx))
            img_cnt += 1
        else:
            print(osp.join(indir, subj, act, imageidx))
    print(img_cnt)

def del_notused_h36mp2_test():
    indir = r"G:\second_model_report_data\datas\three_dimension_reconstruction\h36m\imgs"
    npzpath = r"G:\second_model_report_data\datas\three_dimension_reconstruction\spin_pymaf_data\dataset_extras\h36m_valid_protocol2.npz"
    outpath = osp.join(r"H:\h36m\imgs")
    if not osp.exists(outpath):
        os.makedirs(outpath)

    imgname = np.load(npzpath, allow_pickle=True)["imgname"]
    print(imgname.shape[0])

    img_cnt = 0
    for i in range(imgname.shape[0]):
        dirs = imgname[i].split("/") # S9/S9_Directions_1.60457274/S9_Directions_1.60457274_000001.jpg
        subj = dirs[0]
        act = dirs[1]
        imageidx = dirs[-1]

        if not osp.exists(osp.join(outpath, subj, act)):
            os.makedirs(osp.join(outpath, subj, act))

        if not osp.exists(osp.join(outpath, subj, act, imageidx)) and osp.exists(osp.join(indir, subj, act, imageidx)):
            shutil.copyfile(osp.join(indir, subj, act, imageidx), osp.join(outpath, subj, act, imageidx))
            img_cnt += 1
        else:
            print(osp.join(indir, subj, act, imageidx))
    print(img_cnt)

def del_notused_h36mp1_test():
    indir = r"G:\second_model_report_data\datas\three_dimension_reconstruction\h36m\imgs"
    npzpath = r"G:\second_model_report_data\datas\three_dimension_reconstruction\spin_pymaf_data\dataset_extras\h36m_valid_protocol1.npz"
    outpath = osp.join(r"H:\h36m\imgs")
    if not osp.exists(outpath):
        os.makedirs(outpath)

    imgname = np.load(npzpath, allow_pickle=True)["imgname"]
    print(imgname.shape[0])

    img_cnt = 0
    for i in range(imgname.shape[0]):
        dirs = imgname[i].split("/") # S9/S9_Directions_1.60457274/S9_Directions_1.60457274_000001.jpg
        subj = dirs[0]
        act = dirs[1]
        imageidx = dirs[-1]

        if not osp.exists(osp.join(outpath, subj, act)):
            os.makedirs(osp.join(outpath, subj, act))

        if not osp.exists(osp.join(outpath, subj, act, imageidx)) and osp.exists(osp.join(indir, subj, act, imageidx)):
            shutil.copyfile(osp.join(indir, subj, act, imageidx), osp.join(outpath, subj, act, imageidx))
            img_cnt += 1
        else:
            print(osp.join(indir, subj, act, imageidx))
    print(img_cnt)

def del_notused_h36_train():
    indir = r"G:\second_model_report_data\datas\three_dimension_reconstruction\h36m\imgs"
    npzpath = r"G:\second_model_report_data\datas\three_dimension_reconstruction\spin_pymaf_data\dataset_extras\h36m_mosh_train.npz"
    outpath = osp.join(r"H:\h36m\imgs")
    if not osp.exists(outpath):
        os.makedirs(outpath)

    imgname = np.load(npzpath, allow_pickle=True)["imgname"]
    print(imgname.shape[0])

    img_cnt = 0
    for i in range(imgname.shape[0]):
        dirs = imgname[i].split("/") # S1/S1_Directions_1.54138969/S1_Directions_1.54138969_000001.jpg
        subj = dirs[0]
        act = dirs[1]
        imageidx = dirs[-1]

        if not osp.exists(osp.join(outpath, subj, act)):
            os.makedirs(osp.join(outpath, subj, act))

        if not osp.exists(osp.join(outpath, subj, act, imageidx)) and osp.exists(osp.join(indir, subj, act, imageidx)):
            shutil.copyfile(osp.join(indir, subj, act, imageidx), osp.join(outpath, subj, act, imageidx))
            img_cnt += 1
        else:
            print(osp.join(indir, subj, act, imageidx))
    print(img_cnt)


if __name__ == '__main__':
    # del_notused_h36mp2mosh_test()
    # del_notused_h36mp2_test()
    # del_notused_h36mp1_test()
    # del_notused_h36_train()
    pass