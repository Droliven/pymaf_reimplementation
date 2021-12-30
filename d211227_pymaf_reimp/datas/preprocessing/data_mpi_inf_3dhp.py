#!/usr/bin/env python
# encoding: utf-8
'''
@project : pymaf_reimp
@file    : data_mpi_inf_3dhp.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-12-30 15:30
'''
import os.path as osp
import numpy as np
import os
import shutil
import cv2

def split_mpi_inf_3dhp_train():
    path = r"G:\second_model_report_data\datas\three_dimension_reconstruction\spin_pymaf_data\dataset_extras\mpi_inf_3dhp_train.npz"
    imgname = np.load(path, allow_pickle=True)["imgname"]
    print(imgname.shape[0])

    user_list = range(1, 9)
    seq_list = range(1, 3)
    vid_list = list(range(3)) + list(range(4, 9))

    img_dict = {f"S{user}": {f"Seq{seq}": {f"video_{video}": [] for video in vid_list} for seq in seq_list} for user in user_list}

    old_user = "S1"
    old_seq = "Seq1"
    old_video = "video_0"

    saved_cnt = 0
    img_cnt = 0
    for i in range(imgname.shape[0]):
        dirs = imgname[i].split("/") # S1 / Seq1 / imageFrames / video_0 / frame_000001.jpg

        S = dirs[0]
        Seq = dirs[1]
        vid = dirs[3]
        imageidx = dirs[-1]
        if vid != old_video:
            save_path = "-".join([old_user, old_seq, old_video, str(saved_cnt)]) + ".txt"
            with open(save_path, "w") as f:
                f.write("\n".join(img_dict[old_user][old_seq][old_video]))

            print(f"{saved_cnt}: {old_user}{old_seq}{old_video}: {len(img_dict[old_user][old_seq][old_video])}")
            img_cnt += len(img_dict[old_user][old_seq][old_video])

            img_dict[old_user][old_seq][old_video] = []
            saved_cnt += 1

        img_dict[S][Seq][vid].append(imageidx)
        old_user = S
        old_seq = Seq
        old_video = vid

        if i == imgname.shape[0] - 1:
            save_path = "-".join([old_user, old_seq, old_video, str(saved_cnt)]) + ".txt"
            with open(save_path, "w") as f:
                f.write("\n".join(img_dict[old_user][old_seq][old_video]))

            print(f"{saved_cnt}: {old_user}{old_seq}{old_video}: {len(img_dict[old_user][old_seq][old_video])}")
            img_cnt += len(img_dict[old_user][old_seq][old_video])

            img_dict[old_user][old_seq][old_video] = []
    print(img_cnt)

def avi2jpg_onlyused_mpi_inf_3dhp_train(subjects):
    # indir = r"G:\second_model_report_data\datas\three_dimension_reconstruction\mpi_inf_3dhp\mpi_inf_3dhp_train_set"
    # # indir = r"H:\mpi_inf_3dhp_train_set"
    # txt_path = r"E:\PythonWorkspace\three_dimension_reconstruction\pymaf_reimp\d211227_pymaf_reimp\datas\txt"
    # outpath = osp.join(r"E:\tmp")

    # indir = r"/mnt/hdd4T/dlw_home/model_report_data/datasets/mpi_inf_3dhp"
    # txt_path = r"/mnt/hdd4T/dlw_home/model_report_data/datasets/mpi_inf_3dhp/txt"
    # outpath = osp.join(r"./out")

    indir = r"/home/ml_group/songbo/danglingwei204/datasets/mpi_inf_3dhp"
    # indir = r"H:\mpi_inf_3dhp_train_set"
    txt_path = r"/home/ml_group/songbo/danglingwei204/datasets/mpi_inf_3dhp/txt"
    outpath = osp.join(r"/home/ml_group/songbo/danglingwei204/datasets/mpi_inf_3dhp")

    txts = os.listdir(txt_path)
    cnt = 0
    for t in txts:  # S1-Seq1-video_1-1.txt
        dirs = t.split("-")
        S = dirs[0]
        Seq = dirs[1]
        video = dirs[2]
        # if S in subjects:
        if S in subjects and Seq == "Seq2":

            with open(osp.join(txt_path, t), "r") as f:
                imgs = f.read()
                imgs = imgs.split("\n")

            # video file
            vid_file = os.path.join(indir, S, Seq, 'imageSequence', video + '.avi')
            vidcap = cv2.VideoCapture(vid_file)

            out_path = os.path.join(outpath, S, Seq, 'imageFrames', video)
            if not os.path.isdir(out_path):
                os.makedirs(out_path)

            # process video
            frame = 0
            while 1:
                # extract all frames
                success, image = vidcap.read()
                if not success:
                    break
                frame += 1
                # image name
                imgname = 'frame_%06d.jpg' % frame
                if imgname in imgs :
                    # save image
                    cv2.imwrite(osp.join(out_path, imgname), image)
                    cnt += 1
            print(f"{S}-{Seq}-{frame}-{cnt}")

def change_mpi_inf_3dhp_eval_npz():
    ds = "mpi_inf_3dhp"
    outputs_path = r"H:\mpi_inf_3dhp"
    if not osp.exists(outputs_path):
        os.makedirs(outputs_path)

    # 顺序遍历检索
    npz_data = np.load(r"G:\second_model_report_data\datas\three_dimension_reconstruction\spin_pymaf_data\dataset_extras\mpi_inf_3dhp_valid.npz", allow_pickle=True)
    print(f"test >>> {ds}: {npz_data.files}, shape: {npz_data['imgname'].shape[0]}",
          {npz_data['imgname'][0]})  # 'imgname', 'center', 'scale', 'part', 'S'

    new_imgname = []
    for i in range(npz_data['imgname'].shape[0]):
        old_v = npz_data['imgname'][i]  # mpi_inf_3dhp_test_set/TS1/imageSequence/img_000001.jpg
        new_v = old_v[22:]
        new_imgname.append(new_v)

    np.savez(os.path.join(outputs_path, "mpi_inf_3dhp_valid.npz"), imgname=np.array(new_imgname), center=npz_data["center"], scale=npz_data["scale"], part=npz_data["part"], S=npz_data["S"])

def split_mpi_inf_3dhp_test():
    path = r"G:\second_model_report_data\datas\three_dimension_reconstruction\spin_pymaf_data\dataset_extras\mpi_inf_3dhp_valid.npz"
    imgname = np.load(path, allow_pickle=True)["imgname"]
    print(imgname.shape[0])

    # G:\second_model_report_data\datas\three_dimension_reconstruction\mpi_inf_3dhp\mpi_inf_3dhp_test_set\TS1\imageSequence
    user_list = range(1, 7)
    img_dict = {f"TS{user}": [] for user in user_list}

    old_user = "TS1"

    saved_cnt = 0
    img_cnt = 0
    for i in range(imgname.shape[0]):
        dirs = imgname[i].split("/") # TS1/imageSequence/img_000001.jpg
        TS = dirs[0]
        imageidx = dirs[-1]

        if TS != old_user:
            save_path = "-".join([old_user, str(saved_cnt)]) + ".txt"
            with open(save_path, "w") as f:
                f.write("\n".join(img_dict[old_user]))

            print(f"{saved_cnt}: {old_user}: {len(img_dict[old_user])}")
            img_cnt += len(img_dict[old_user])

            img_dict[old_user] = []
            saved_cnt += 1

        img_dict[TS].append(imageidx)
        old_user = TS

        if i == imgname.shape[0] - 1:
            save_path = "-".join([old_user, str(saved_cnt)]) + ".txt"
            with open(save_path, "w") as f:
                f.write("\n".join(img_dict[old_user]))

            print(f"{saved_cnt}: {old_user}: {len(img_dict[old_user])}")
            img_cnt += len(img_dict[old_user])

            img_dict[old_user] = []
    print(img_cnt)

def del_notused_mpi_inf_3dhp_test():
    indir = r"G:\second_model_report_data\datas\three_dimension_reconstruction\mpi_inf_3dhp\mpi_inf_3dhp_test_set"
    txt_path = r"E:\PythonWorkspace\three_dimension_reconstruction\pymaf_reimp\d211227_pymaf_reimp\datas\txt"
    outpath = osp.join(r"H:\mpi_inf_3dhp\mpi_inf_3dhp_test_set")

    txt_list = [f"TS{user}-{user-1}.txt" for user in range(1, 7)]

    cnt = 0
    for t in txt_list: # TS1-0.txt
        S = t.split("-")[0]
        with open(osp.join(txt_path, t), "r") as f:
            imgs = f.read()
            imgs = imgs.split("\n")

        if not osp.exists(osp.join(outpath, S, "imageSequence")):
            os.makedirs(osp.join(outpath, S, "imageSequence"))

        for im in imgs:
            if osp.exists(osp.join(indir, S, "imageSequence", im)):
                shutil.copyfile(osp.join(indir, S, "imageSequence", im), osp.join(outpath, S, "imageSequence", im))
                cnt += 1
            else:
                print(osp.join(indir, S, "imageSequence", im))
    print(cnt)

if __name__ == '__main__':
    # split_mpi_inf_3dhp_train()

    # avi2jpg_onlyused_mpi_inf_3dhp_train(subjects=["S1"]) # 13499, 7.04G
    # avi2jpg_onlyused_mpi_inf_3dhp_train(subjects=["S2"]) # 8785, 4.78G
    # avi2jpg_onlyused_mpi_inf_3dhp_train(subjects=["S3"]) # 17928, 8.60G
    # avi2jpg_onlyused_mpi_inf_3dhp_train(subjects=["S4"]) # 9607, 5.50G
    # avi2jpg_onlyused_mpi_inf_3dhp_train(subjects=["S5"]) # 18703, 8.98G
    # avi2jpg_onlyused_mpi_inf_3dhp_train(subjects=["S6"]) # 9392 4.88G
    # avi2jpg_onlyused_mpi_inf_3dhp_train(subjects=["S7"]) # 9718, 4.82G
    # avi2jpg_onlyused_mpi_inf_3dhp_train(subjects=["S8"]) # 8875, 4.70G
    # change_mpi_inf_3dhp_eval_npz()
    # split_mpi_inf_3dhp_test()
    # del_notused_mpi_inf_3dhp_test()
    pass
