#!/usr/bin/env python
# encoding: utf-8
'''
@project : pymaf_reimp
@file    : regressor.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2022-01-03 21:18
'''

import torch
import torch.nn as nn
import numpy as np

from .smpl import SMPL
from ..utils.geometry import rot6d_to_rotmat, projection, rotation_matrix_to_angle_axis


class Regressor(nn.Module):
    def __init__(self, cfg, feat_dim):
        super(Regressor, self).__init__()

        self.cfg = cfg
        self.H36M_TO_J14 = self.cfg.constants.h36m_to_j14

        npose = 24 * 6

        self.fc1 = nn.Linear(feat_dim + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        self.smpl = SMPL(cfg, self.cfg.run.smpl_model_path, batch_size=cfg.train.batch_size, create_transl=False)

        mean_params = np.load(self.cfg.run.smpl_mean_params_path)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0) # b, 1, 144
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=1, J_regressor=None):
        '''

        :param x: [b, 2205/2155]
        :param init_pose: [b, 144]
        :param init_shape: [b, 10]
        :param init_cam: [b, 3]
        :param n_iter: 1
        :param J_regressor:
        :return:
        '''
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1) # todo 这种 是 txy, 还是 xyz

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose # 显式残差
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False
        )

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints
        pred_smpl_joints = pred_output.smpl_joints
        pred_keypoints_2d = projection(pred_joints, pred_cam) # todo：这里有两个问题，第一 cam 是sxy 还是 xyz? 第二 不同尺度 img_size 都设 224 是不是不合适
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        if J_regressor is not None:
            pred_joints = torch.matmul(J_regressor, pred_vertices)
            pred_pelvis = pred_joints[:, [0], :].clone()
            pred_joints = pred_joints[:, self.H36M_TO_J14, :]
            pred_joints = pred_joints - pred_pelvis

        output = {
            'theta'  : torch.cat([pred_cam, pred_shape, pose], dim=1),
            'verts'  : pred_vertices,
            # 'kp_2d'  : pred_keypoints_2d,
            # 'kp_3d'  : pred_joints,
            # 'smpl_kp_3d' : pred_smpl_joints,
            'rotmat' : pred_rotmat,
            'pred_pose': pred_pose,
        }
        return output

    def forward_init(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=1, J_regressor=None):
        # [b, 2048]
        batch_size = x.shape[0]
        # todo 为什么这里 init_pose 是 144, 因为用的是 rot6d 表示法
        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1) # [1, 144] -> [b, 144]
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1) # [1, 10] -> [b, 10]
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1) # [1, 3] -> [b, 3]

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose.contiguous()).view(batch_size, 24, 3, 3)

        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False
        )
        # {vertices = vertices,  # [b, 6890, 3]
        # global_orient = smpl_output.global_orient,  # [b, 1, 3, 3]
        # body_pose = smpl_output.body_pose,  # [b, 23, 3, 3]
        # joints = joints,  # [b, 49, 3]
        # joints_J19 = joints_J19,  # [b, 19, 3]
        # smpl_joints = smpl_joints,  # [b, 24, 3]
        # betas = smpl_output.betas,  # [b, 10]
        # full_pose = smpl_output.full_pose)}  # NONE

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints
        pred_smpl_joints = pred_output.smpl_joints
        pred_keypoints_2d = projection(pred_joints, pred_cam)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        if J_regressor is not None:
            pred_joints = torch.matmul(J_regressor, pred_vertices)
            pred_pelvis = pred_joints[:, [0], :].clone()
            pred_joints = pred_joints[:, self.H36M_TO_J14, :]
            pred_joints = pred_joints - pred_pelvis

        output = {
            'theta'  : torch.cat([pred_cam, pred_shape, pose], dim=1),
            'verts'  : pred_vertices,
            # 'kp_2d'  : pred_keypoints_2d,
            # 'kp_3d'  : pred_joints,
            # 'smpl_kp_3d' : pred_smpl_joints,
            'rotmat' : pred_rotmat,
            'pred_pose': pred_pose,
        }
        return output
