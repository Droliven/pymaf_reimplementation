#!/usr/bin/env python
# encoding: utf-8
'''
@project : pymaf_reimp
@file    : mixed_dataset.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-12-27 20:07
'''
"""
# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/datasets/mixed_dataset.py
This file contains the definition of different heterogeneous datasets used for training
"""
import torch
import numpy as np

from .base_dataset import BaseDataset

class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, eval_pve, noise_factor, rot_factor, scale_factor, ignore_3d, use_augmentation, is_train, is_debug, DATASET_FOLDERS, DATASET_FILES, JOINT_MAP, JOINT_NAMES, J24_TO_J19, JOINT_REGRESSOR_TRAIN_EXTRA, SMPL_MODEL_DIR, IMG_NORM_MEAN, IMG_NORM_STD, TRAIN_BATCH_SIZE, IMG_RES, SMPL_JOINTS_FLIP_PERM, SMPL_POSE_FLIP_PERM):
        self.dataset_list = ['h36m', 'lsp_orig', 'mpii', 'lspet', 'coco', 'mpi_inf_3dhp']
        self.dataset_dict = {'h36m': 0, 'lsp_orig': 1, 'mpii': 2, 'lspet': 3, 'coco': 4, 'mpi_inf_3dhp': 5}

        self.datasets = [BaseDataset(eval_pve, noise_factor, rot_factor, scale_factor, ds, ignore_3d, use_augmentation, is_train, is_debug, DATASET_FOLDERS, DATASET_FILES, JOINT_MAP, JOINT_NAMES, J24_TO_J19, JOINT_REGRESSOR_TRAIN_EXTRA, SMPL_MODEL_DIR, IMG_NORM_MEAN, IMG_NORM_STD, TRAIN_BATCH_SIZE, IMG_RES, SMPL_JOINTS_FLIP_PERM, SMPL_POSE_FLIP_PERM) for ds in self.dataset_list]
        self.dataset_length = {self.dataset_list[idx]: len(ds) for idx, ds in enumerate(self.datasets)}
        length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
        self.length = max([len(ds) for ds in self.datasets]) # 为什么设定一个最大长度
        """
        Data distribution inside each batch:
        30% H36M - 60% ITW - 10% MPI-INF
        """
        self.partition = [
                            .3,
                            .6*len(self.datasets[1])/length_itw,
                            .6*len(self.datasets[2])/length_itw,
                            .6*len(self.datasets[3])/length_itw,
                            .6*len(self.datasets[4])/length_itw,
                            0.1]
        self.partition = np.array(self.partition).cumsum()

    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(6):
            if p <= self.partition[i]:
                return self.datasets[i][index % len(self.datasets[i])] # 根据最大长度循环取余

    def __len__(self):
        return self.length
