#!/usr/bin/env python
# encoding: utf-8
'''
@project : pymaf_reimp
@file    : mixed_train_ds.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2022-01-21 10:09
'''
"""
# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/datasets/mixed_dataset.py
This file contains the definition of different heterogeneous datasets used for training
"""
import torch
import numpy as np

from .base_ds import BaseDS

class MixedTrainDS(torch.utils.data.Dataset):

    def __init__(self, is_debug, cfg, ignore_3d = False, use_augmentation = True, is_train = True):

        self.dataset_list = ['train_h36m', 'train_lsporig', 'train_mpii', 'train_lspet', 'train_coco2014', 'train_mpiinf3dhp']
        self.dataset_dict = {'train_h36m': 0, 'train_lsporig': 1, 'train_mpii': 2, 'train_lspet': 3, 'train_coco2014': 4, 'train_mpiinf3dhp': 5}

        self.datasets = [BaseDS(is_debug, cfg, data_cfg, ignore_3d = ignore_3d, use_augmentation = use_augmentation, is_train = is_train) for data_cfg in cfg.dataset.train_list]

        # total_length = sum([len(ds) for ds in self.datasets])
        length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
        self.length = max([len(ds) for ds in self.datasets])
        """
        Data distribution inside each batch:
        30% H36M - 60% ITW - 10% MPI-INF
        """
        self.partition = [.3, .6*len(self.datasets[1])/length_itw,
                          .6*len(self.datasets[2])/length_itw,
                          .6*len(self.datasets[3])/length_itw,
                          .6*len(self.datasets[4])/length_itw,
                          0.1]
        self.partition = np.array(self.partition).cumsum()

    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(6):
            if p <= self.partition[i]:
                return self.datasets[i][index % len(self.datasets[i])]

    def __len__(self):
        return self.length