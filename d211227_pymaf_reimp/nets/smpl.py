#!/usr/bin/env python
# encoding: utf-8
'''
@project : pymaf_reimp
@file    : smpl.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-12-27 16:35
'''
# This script is borrowed from https://github.com/nkolot/SPIN/blob/master/models/smpl.py

import torch
import numpy as np
from smplx import SMPL as _SMPL
from smplx.body_models import ModelOutput
from smplx.lbs import vertices2joints
from collections import namedtuple
import os.path as osp

class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cfg = cfg

        joints = [cfg.constants.joint_map[i] for i in cfg.constants.joint_names]
        J_regressor_extra = np.load(cfg.run.J_regressor_train_extra_path)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)
        self.ModelOutput = namedtuple('ModelOutput_', ModelOutput._fields + ('smpl_joints', 'joints_J19',))
        self.ModelOutput.__new__.__defaults__ = (None,) * len(self.ModelOutput._fields)

    def forward(self, *args, **kwargs):
        '''

        :param args:
        :param kwargs: {'betas': [b, 10], 'body_pose': [b, 23, 3, 3], 'global_orient': [b, 1, 3, 3], 'pose2rot': False}
        :return:
        '''
        kwargs['get_skin'] = True
        smpl_output = super().forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices) # [9, 6890] * [b, 6890, 3] -> [b, 9, 3]
        # smpl_output.joints: [B, 45, 3]  extra_joints: [B, 9, 3]
        vertices = smpl_output.vertices
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1) # b, 45+9, 3
        smpl_joints = smpl_output.joints[:, :24] # [b, 24, 3] todo 为什么前 24个点 是这个意思，但总共有 45点
        joints = joints[:, self.joint_map, :]   # [B, 49, 3] todo 为什么从 54 个取出来 49个
        joints_J24 = joints[:, -24:, :] # 为什么一会 24 一会儿 19
        joints_J19 = joints_J24[:, self.cfg.constants.j24_to_j19, :]
        output = self.ModelOutput(vertices=vertices, # [b, 6890, 3]
                                  global_orient=smpl_output.global_orient, # [b, 1, 3, 3]
                                  body_pose=smpl_output.body_pose, # [b, 23, 3, 3]
                                  joints=joints, # [b, 49, 3]
                                  joints_J19=joints_J19, # [b, 19, 3]
                                  smpl_joints=smpl_joints, # [b, 24, 3]
                                  betas=smpl_output.betas, # [b, 10]
                                  full_pose=smpl_output.full_pose) # NONE
        return output

def get_smpl_faces(JOINT_MAP, JOINT_NAMES, J24_TO_J19, JOINT_REGRESSOR_TRAIN_EXTRA, SMPL_MODEL_DIR):
    smpl = SMPL(JOINT_MAP, JOINT_NAMES, J24_TO_J19, JOINT_REGRESSOR_TRAIN_EXTRA, SMPL_MODEL_DIR, batch_size=1, create_transl=False)
    return smpl.faces

def get_part_joints(smpl_joints):
    batch_size = smpl_joints.shape[0]

    # part_joints = torch.zeros().to(smpl_joints.device)

    one_seg_pairs = [(0, 1), (0, 2), (0, 3), (3, 6), (9, 12), (9, 13), (9, 14), (12, 15), (13, 16), (14, 17)]
    two_seg_pairs = [(1, 4), (2, 5), (4, 7), (5, 8), (16, 18), (17, 19), (18, 20), (19, 21)]

    one_seg_pairs.extend(two_seg_pairs)

    single_joints = [(10), (11), (15), (22), (23)]

    part_joints = []

    for j_p in one_seg_pairs:
        new_joint = torch.mean(smpl_joints[:, j_p], dim=1, keepdim=True)
        part_joints.append(new_joint)

    for j_p in single_joints:
        part_joints.append(smpl_joints[:, j_p:j_p+1])

    part_joints = torch.cat(part_joints, dim=1)

    return part_joints
