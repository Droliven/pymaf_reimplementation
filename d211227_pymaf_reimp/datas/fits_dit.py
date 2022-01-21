#!/usr/bin/env python
# encoding: utf-8
'''
@project : pymaf_reimp
@file    : fits_dit.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-12-27 21:27
'''

'''
This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/train/fits_dict.py
'''
import os
import cv2
import torch
import numpy as np
from torchgeometry import angle_axis_to_rotation_matrix


class FitsDict():
    """ Dictionary keeping track of the best fit per image in the training set """
    def __init__(self, is_single_dataset, output_dir, train_dataset, is_debug, SMPL_POSE_FLIP_PERM, FINAL_FITS_DIR):
        self.is_debug = is_debug

        self.output_dir = output_dir
        self.train_dataset = train_dataset
        self.fits_dict = {}  # 预存好的伪真值
        self.valid_fit_state = {} # todo 这玩意儿代表啥
        # array used to flip SMPL pose parameters
        self.flipped_parts = torch.tensor(SMPL_POSE_FLIP_PERM, dtype=torch.int64)
        # Load dictionary state
        for ds_name, ds in train_dataset.dataset_dict.items():
            if ds_name in ['h36m']:
                dict_file = os.path.join(FINAL_FITS_DIR, ds_name + '.npy')
                fits_dict = torch.from_numpy(np.load(dict_file))
                valid_fit_state = torch.ones(len(fits_dict), dtype=torch.uint8)
            else:
                dict_file = os.path.join(FINAL_FITS_DIR, ds_name + '.npz')
                fits_dict = np.load(dict_file)
                opt_pose = torch.from_numpy(fits_dict['pose'])
                opt_betas = torch.from_numpy(fits_dict['betas'])
                opt_valid_fit = torch.from_numpy(fits_dict['valid_fit']).to(torch.uint8)
                fits_dict = torch.cat([opt_pose, opt_betas], dim=1)
                valid_fit_state = opt_valid_fit

            if self.is_debug:
                self.fits_dict[ds_name] = fits_dict[:20000]
                self.valid_fit_state[ds_name] = valid_fit_state[:20000]
            else:
                self.fits_dict[ds_name] = fits_dict
                self.valid_fit_state[ds_name] = valid_fit_state


        # todo 这一段操作会更改传入的 train dataset
        if not is_single_dataset:
            for ds in train_dataset.datasets:
                if ds.dataset not in ['h36m']:
                    ds.pose = self.fits_dict[ds.dataset][:, :72].numpy()
                    ds.shape = self.fits_dict[ds.dataset][:, 72:].numpy()
                    ds.has_smpl = self.valid_fit_state[ds.dataset].numpy()

    def save(self):
        """ Save dictionary state to disk """
        for ds_name in self.train_dataset.dataset_dict.keys():
            dict_file = os.path.join(self.output_dir, ds_name + '_fits.npy')
            np.save(dict_file, self.fits_dict[ds_name].cpu().numpy())

    def __getitem__(self, x):
        """ Retrieve dictionary entries """
        dataset_name, ind, rot, is_flipped = x
        batch_size = len(dataset_name)
        pose = torch.zeros((batch_size, 72))
        betas = torch.zeros((batch_size, 10))
        for ds, i, n in zip(dataset_name, ind, range(batch_size)):
            params = self.fits_dict[ds][i]
            pose[n, :] = params[:72]
            betas[n, :] = params[72:]
        pose = pose.clone()
        # Apply flipping and rotation
        pose = self.flip_pose(self.rotate_pose(pose, rot), is_flipped)
        betas = betas.clone()
        return pose, betas

    def get_vaild_state(self, dataset_name, ind):
        batch_size = len(dataset_name)
        valid_fit = torch.zeros(batch_size, dtype=torch.uint8)
        for ds, i, n in zip(dataset_name, ind, range(batch_size)):
            valid_fit[n] = self.valid_fit_state[ds][i]
        valid_fit = valid_fit.clone()
        return valid_fit

    def __setitem__(self, x, val):
        """ Update dictionary entries """
        dataset_name, ind, rot, is_flipped, update = x
        pose, betas = val
        batch_size = len(dataset_name)
        # Undo flipping and rotation
        pose = self.rotate_pose(self.flip_pose(pose, is_flipped), -rot)
        params = torch.cat((pose, betas), dim=-1).cpu()
        for ds, i, n in zip(dataset_name, ind, range(batch_size)):
            if update[n]:
                self.fits_dict[ds][i] = params[n]

    def flip_pose(self, pose, is_flipped):
        """flip SMPL pose parameters"""
        # is_flipped = is_flipped.byte()
        is_flipped = is_flipped.bool()
        pose_f = pose.clone()
        pose_f[is_flipped, :] = pose[is_flipped][:, self.flipped_parts]
        # we also negate the second and the third dimension of the axis-angle representation
        pose_f[is_flipped, 1::3] *= -1
        pose_f[is_flipped, 2::3] *= -1
        return pose_f

    def rotate_pose(self, pose, rot):
        """Rotate SMPL pose parameters by rot degrees"""
        pose = pose.clone()
        cos = torch.cos(-np.pi * rot / 180.)
        sin = torch.sin(-np.pi * rot/ 180.)
        zeros = torch.zeros_like(cos)
        r3 = torch.zeros(cos.shape[0], 1, 3, device=cos.device)
        r3[:,0,-1] = 1
        R = torch.cat([torch.stack([cos, -sin, zeros], dim=-1).unsqueeze(1),
                       torch.stack([sin, cos, zeros], dim=-1).unsqueeze(1),
                       r3], dim=1)
        global_pose = pose[:, :3]
        global_pose_rotmat = angle_axis_to_rotation_matrix(global_pose)
        global_pose_rotmat_3b3 = global_pose_rotmat[:, :3, :3]
        global_pose_rotmat_3b3 = torch.matmul(R, global_pose_rotmat_3b3)
        global_pose_rotmat[:, :3, :3] = global_pose_rotmat_3b3
        global_pose_rotmat = global_pose_rotmat[:, :-1, :-1].cpu().numpy()
        global_pose_np = np.zeros((global_pose.shape[0], 3))
        for i in range(global_pose.shape[0]):
            aa, _ = cv2.Rodrigues(global_pose_rotmat[i])
            global_pose_np[i,:] = aa.squeeze()
        pose[:, :3] = torch.from_numpy(global_pose_np).to(pose.device)
        return pose
