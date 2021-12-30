#!/usr/bin/env python
# encoding: utf-8
'''
@project : pymaf_reimp
@file    : data_coco.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-12-30 20:34
'''

import os.path as osp
import numpy as np
import os
import shutil
import cv2

def del_notused_coco_test():
    indir = r"G:\second_model_report_data\datas\three_dimension_reconstruction\coco\val2014"
    npzpath = r"G:\second_model_report_data\datas\three_dimension_reconstruction\spin_pymaf_data\dataset_extras\coco_2014_val.npz"
    outpath = osp.join(r"H:\coco\val2014")
    if not osp.exists(outpath):
        os.makedirs(outpath)

    imgname = np.load(npzpath, allow_pickle=True)["imgname"]
    print(imgname.shape[0])

    img_cnt = 0
    for i in range(imgname.shape[0]):
        imageidx = imgname[i].split("/")[-1]  # val2014/COCO_val2014_000000537548.jpg
        if not osp.exists(osp.join(outpath, imageidx)) and osp.exists(osp.join(indir, imageidx)):
            shutil.copyfile(osp.join(indir, imageidx), osp.join(outpath, imageidx))
            img_cnt += 1
        else:
            print(osp.join(indir, imageidx))
    print(img_cnt)

def del_notused_coco_train():
    indir = r"G:\second_model_report_data\datas\three_dimension_reconstruction\coco\train2014"
    npzpath = r"G:\second_model_report_data\datas\three_dimension_reconstruction\spin_pymaf_data\dataset_extras\coco_2014_train.npz"
    outpath = osp.join(r"H:\coco\train2014")
    if not osp.exists(outpath):
        os.makedirs(outpath)

    imgname = np.load(npzpath, allow_pickle=True)["imgname"]
    print(imgname.shape[0])

    img_cnt = 0
    for i in range(imgname.shape[0]):
        imageidx = imgname[i].split("/")[-1]  # train2014/COCO_val2014_000000537548.jpg
        if not osp.exists(osp.join(outpath, imageidx)) and osp.exists(osp.join(indir, imageidx)):
            shutil.copyfile(osp.join(indir, imageidx), osp.join(outpath, imageidx))
            img_cnt += 1
        else:
            print(osp.join(indir, imageidx))
    print(img_cnt)

if __name__ == '__main__':
    # del_notused_coco_test()
    # del_notused_coco_train()
    pass