#!/usr/bin/env python
# encoding: utf-8
'''
@project : pymaf_reimp
@file    : data_3dpw.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-12-30 20:18
'''

import os.path as osp
import numpy as np
import os
import shutil
import cv2


def split_3dpw_test():
    path = r"G:\second_model_report_data\datas\three_dimension_reconstruction\spin_pymaf_data\dataset_extras\3dpw_test.npz"
    out_dir = osp.abspath("./txt")
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    imgname = np.load(path, allow_pickle=True)["imgname"]
    print(imgname.shape[0])

    # G:\second_model_report_data\datas\three_dimension_reconstruction\3dpw\3dpw_test_set\TS1\imageSequence
    act_img_dict = {}
    old_act = "downtown_enterShop_00"

    saved_cnt = 0
    img_cnt = 0
    for i in range(imgname.shape[0]):
        dirs = imgname[i].split("/") # imageFiles/downtown_enterShop_00/image_00000.jpg
        act = dirs[1]
        imageidx = dirs[-1]
        if act not in act_img_dict:
            act_img_dict[act] = []

        if act != old_act:
            save_path = osp.join(out_dir, "-".join([old_act, str(saved_cnt)]) + ".txt")
            with open(save_path, "w") as f:
                f.write("\n".join(act_img_dict[old_act]))

            print(f"{saved_cnt}: {old_act}: {len(act_img_dict[old_act])}")
            img_cnt += len(act_img_dict[old_act])

            act_img_dict[old_act] = []
            saved_cnt += 1

        act_img_dict[act].append(imageidx)
        old_act = act

        if i == imgname.shape[0] - 1:
            save_path = osp.join(out_dir, "-".join([old_act, str(saved_cnt)]) + ".txt")
            with open(save_path, "w") as f:
                f.write("\n".join(act_img_dict[old_act]))

            print(f"{saved_cnt}: {old_act}: {len(act_img_dict[old_act])}")
            img_cnt += len(act_img_dict[old_act])

            act_img_dict[old_act] = []
    print(img_cnt)

def del_notused_3dpw_test():
    indir = r"G:\second_model_report_data\datas\three_dimension_reconstruction\3dpw\imageFiles"
    txt_path = r"E:\PythonWorkspace\three_dimension_reconstruction\pymaf_reimp\d211227_pymaf_reimp\datas\txt"
    outpath = osp.join(r"H:\3dpw\imageFiles")

    txt_list = os.listdir(txt_path)

    cnt = 0
    for t in txt_list: # downtown_arguing_00-15.txt
        act = t.split("-")[0]
        with open(osp.join(txt_path, t), "r") as f:
            imgs = f.read()
            imgs = imgs.split("\n")

        if not osp.exists(osp.join(outpath, act)):
            os.makedirs(osp.join(outpath, act))

        for im in imgs:
            if not osp.exists(osp.join(outpath, act, im)) and osp.exists(osp.join(indir, act, im)):
                shutil.copyfile(osp.join(indir, act, im), osp.join(outpath, act, im))
                cnt += 1
            else:
                print(osp.join(indir, act, im))
    print(cnt)


if __name__ == '__main__':
    # split_3dpw_test()
    # del_notused_3dpw_test()
    pass