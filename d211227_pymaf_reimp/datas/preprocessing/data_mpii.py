#!/usr/bin/env python
# encoding: utf-8
'''
@project : pymaf_reimp
@file    : data_mpii.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-12-30 21:21
'''

import os.path as osp
import numpy as np
import os
import shutil
import cv2

def del_notused_mpii_train():
    indir = r"G:\second_model_report_data\datas\three_dimension_reconstruction\mpii\images"
    npzpath = r"G:\second_model_report_data\datas\three_dimension_reconstruction\spin_pymaf_data\dataset_extras\mpii_train.npz"
    outpath = osp.join(r"H:\mpii\images")
    if not osp.exists(outpath):
        os.makedirs(outpath)

    imgname = np.load(npzpath, allow_pickle=True)["imgname"]
    print(imgname.shape[0])

    img_cnt = 0
    for i in range(imgname.shape[0]):
        imageidx = imgname[i].split("/")[-1] # images/015601864.jpg
        if not osp.exists(osp.join(outpath, imageidx)) and osp.exists(osp.join(indir, imageidx)):
            shutil.copyfile(osp.join(indir, imageidx), osp.join(outpath, imageidx))
            img_cnt += 1
        else:
            print(f"source: {osp.exists(osp.join(indir, imageidx))}, dist: {osp.exists(osp.join(outpath, imageidx))}; {osp.join(indir, imageidx)}")
    print(img_cnt)

if __name__ == '__main__':
    # del_notused_mpii_train()
    pass