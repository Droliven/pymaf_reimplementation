#!/usr/bin/env python
# encoding: utf-8
'''
@project : pymaf_reimp
@file    : data_lsp.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-12-30 20:43
'''

import os.path as osp
import numpy as np
import os
import shutil
import cv2

def del_notused_lspsmall_test():
    indir = r"G:\second_model_report_data\datas\three_dimension_reconstruction\lsp\lsp_dataset_small\images"
    npzpath = r"G:\second_model_report_data\datas\three_dimension_reconstruction\spin_pymaf_data\dataset_extras\lsp_dataset_test.npz"
    outpath = osp.join(r"H:\lsp\lspsmall\images")
    if not osp.exists(outpath):
        os.makedirs(outpath)

    imgname = np.load(npzpath, allow_pickle=True)["imgname"]
    print(imgname.shape[0])

    img_cnt = 0
    for i in range(imgname.shape[0]):
        imageidx = imgname[i].split("/")[-1]  # images/im1001.jpg
        if not osp.exists(osp.join(outpath, imageidx)) and osp.exists(osp.join(indir, imageidx)):
            shutil.copyfile(osp.join(indir, imageidx), osp.join(outpath, imageidx))
            img_cnt += 1
        else:
            print(osp.join(indir, imageidx))
    print(img_cnt)

def del_notused_lspet_train():
    indir = r"G:\second_model_report_data\datas\three_dimension_reconstruction\lsp\hr_lspet"
    npzpath = r"G:\second_model_report_data\datas\three_dimension_reconstruction\spin_pymaf_data\dataset_extras\hr_lspet_train.npz"
    outpath = osp.join(r"H:\lsp\hr_lspet")
    if not osp.exists(outpath):
        os.makedirs(outpath)

    imgname = np.load(npzpath, allow_pickle=True)["imgname"]
    print(imgname.shape[0])

    img_cnt = 0
    for i in range(imgname.shape[0]):
        imageidx = imgname[i] # im00001.png
        if not osp.exists(osp.join(outpath, imageidx)) and osp.exists(osp.join(indir, imageidx)):
            shutil.copyfile(osp.join(indir, imageidx), osp.join(outpath, imageidx))
            img_cnt += 1
        else:
            print(osp.join(indir, imageidx))
    print(img_cnt)

def del_notused_lsporigin_train():
    indir = r"G:\second_model_report_data\datas\three_dimension_reconstruction\lsp\lsp_dataset_original\images"
    npzpath = r"G:\second_model_report_data\datas\three_dimension_reconstruction\spin_pymaf_data\dataset_extras\lsp_dataset_original_train.npz"
    outpath = osp.join(r"H:\lsp\lsp_dataset_original\images")
    if not osp.exists(outpath):
        os.makedirs(outpath)

    imgname = np.load(npzpath, allow_pickle=True)["imgname"]
    print(imgname.shape[0])

    img_cnt = 0
    for i in range(imgname.shape[0]):
        imageidx = imgname[i].split("/")[-1] # images/im0001.jpg
        if not osp.exists(osp.join(outpath, imageidx)) and osp.exists(osp.join(indir, imageidx)):
            shutil.copyfile(osp.join(indir, imageidx), osp.join(outpath, imageidx))
            img_cnt += 1
        else:
            print(osp.join(indir, imageidx))
    print(img_cnt)

if __name__ == '__main__':
    # del_notused_lspsmall_test()
    # del_notused_lspet_train()
    # del_notused_lsporigin_train()
    pass