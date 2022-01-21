#!/usr/bin/env python
# encoding: utf-8
'''
@project : pymaf_reimp
@file    : pymaf.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-12-27 16:03
'''
import torch
import torch.nn as nn
import numpy as np

from .backbone_resnet import BackboneResNet
from .maf_extractor import MAF_Extractor
from .res_module import IUV_predict_layer
from .regressor import Regressor

class PyMAF(nn.Module):
    """ PyMAF based Deep Regressor for Human Mesh Recovery
    PyMAF: 3D Human Pose and Shape Regression with Pyramidal Mesh Alignment Feedback Loop, in ICCV, 2021
    """

    def __init__(self, cfg, pretrained=True):
        super(PyMAF, self).__init__()

        self.cfg = cfg

        self.feature_extractor = BackboneResNet(model=cfg.train.backbone.name, pretrained=pretrained)

        # deconv layers
        self.inplanes = self.feature_extractor.inplanes
        self.deconv_with_bias = cfg.train.res_model.deconv_with_bias
        self.deconv_layers = self._make_deconv_layer(
            cfg.train.res_model.num_deconv_layers,
            cfg.train.res_model.num_deconv_filters,
            cfg.train.res_model.num_deconv_kernels,
        )

        self.maf_extractor = nn.ModuleList()
        for _ in range(cfg.train.backbone.n_iter):
            self.maf_extractor.append(MAF_Extractor(device=cfg.run.device, data_dir=cfg.run.preprocessed_data_dir, MLP_DIM=cfg.train.backbone.mlp_dim))
        ma_feat_len = self.maf_extractor[-1].Dmap.shape[0] * cfg.train.backbone.mlp_dim[-1]

        grid_size = 21
        xv, yv = torch.meshgrid([torch.linspace(-1, 1, grid_size), torch.linspace(-1, 1, grid_size)])
        points_grid = torch.stack([xv.reshape(-1), yv.reshape(-1)]).unsqueeze(0)
        self.register_buffer('points_grid', points_grid)
        grid_feat_len = grid_size * grid_size * cfg.train.backbone.mlp_dim[-1]

        self.regressor = nn.ModuleList()
        for i in range(cfg.train.backbone.n_iter):
            if i == 0:
                ref_infeat_dim = grid_feat_len  # 2205
            else:
                ref_infeat_dim = ma_feat_len  # 2155
            self.regressor.append(Regressor(cfg, feat_dim=ref_infeat_dim))  # 预训练模型里面是：[6890, 3, 10], 这里是 [6890, 3, 300]

        dp_feat_dim = 256
        self.with_uv = cfg.train.point_regression_weights > 0
        if cfg.train.backbone.aux_supv_on:
            self.dp_head = IUV_predict_layer(feat_dim=dp_feat_dim)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """
        Deconv_layer used in Simple Baselines:
        Xiao et al. Simple Baselines for Human Pose Estimation and Tracking
        https://github.com/microsoft/human-pose-estimation.pytorch
        """
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        def _get_deconv_cfg(deconv_kernel, index):
            if deconv_kernel == 4:
                padding = 1
                output_padding = 0
            elif deconv_kernel == 3:
                padding = 1
                output_padding = 1
            elif deconv_kernel == 2:
                padding = 0
                output_padding = 0

            return deconv_kernel, padding, output_padding

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = _get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=self.cfg.train.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x, J_regressor=None):
        '''

        :param x: image [b, 3, 224, 224]
        :param J_regressor:
        :return:
        '''

        batch_size = x.shape[0]

        # spatial features and global features, 为什么经过池化之后的特征叫 global ?
        s_feat, g_feat = self.feature_extractor(x) # [b, 2048, 7, 7], [b, 2048]

        assert self.cfg.train.backbone.n_iter >= 0 and self.cfg.train.backbone.n_iter <= 3
        if self.cfg.train.backbone.n_iter == 1:
            deconv_blocks = [self.deconv_layers]
        elif self.cfg.train.backbone.n_iter == 2:
            deconv_blocks = [self.deconv_layers[0:6], self.deconv_layers[6:9]]
        elif self.cfg.train.backbone.n_iter == 3:
            deconv_blocks = [self.deconv_layers[0:3], self.deconv_layers[3:6], self.deconv_layers[6:9]]

        out_list = {}

        # initial parameters
        # TODO: remove the initial mesh generation during forward to reduce runtime
        # by generating initial mesh the beforehand: smpl_output = self.init_smpl
        smpl_output = self.regressor[0].forward_init(g_feat, J_regressor=J_regressor)
        # smpl_output = {
        #             'theta'  : [b, 85],
        #             'verts'  : [b, 6890, 3],
        #             'kp_2d'  : [b, 49, 2],
        #             'kp_3d'  : [b, 49, 3]
        #             'smpl_kp_3d' : [b, 24, 3]
        #             'rotmat' : [b, 24, 3, 3],
        #             'pred_cam': [b, 3],
        #             'pred_shape': [b, 10],
        #             'pred_pose': [b, 144],
        #         }

        out_list['smpl_out'] = [smpl_output]
        out_list['dp_out'] = []

        # for visulization
        vis_feat_list = [s_feat.detach()]

        # parameter predictions
        for rf_i in range(self.cfg.train.backbone.n_iter):
            # pred_cam = smpl_output['pred_cam'] # [b, 3]
            # pred_shape = smpl_output['pred_shape'] # [b, 10]
            pred_pose = smpl_output['pred_pose'] # [b, 144]
            pred_cam = smpl_output['theta'][:, :3]  # [b, 3]
            pred_shape = smpl_output['theta'][:, 3:13]  # [b, 10]


            pred_cam = pred_cam.detach()
            pred_shape = pred_shape.detach()
            pred_pose = pred_pose.detach()

            s_feat_i = deconv_blocks[rf_i](s_feat) # [b, 2048, 7, 7] -> [b, 256, 14, 14]
            s_feat = s_feat_i
            vis_feat_list.append(s_feat_i.detach())

            self.maf_extractor[rf_i].im_feat = s_feat_i
            self.maf_extractor[rf_i].cam = pred_cam

            if rf_i == 0:
                sample_points = torch.transpose(self.points_grid.expand(batch_size, -1, -1), 1, 2) # [b, 441, 2]
                ref_feature = self.maf_extractor[rf_i].sampling(sample_points)  # [b, 2205]
            else:
                pred_smpl_verts = smpl_output['verts'].detach()
                # TODO: use a more sparse SMPL implementation (with 431 vertices) for acceleration
                pred_smpl_verts_ds = torch.matmul(self.maf_extractor[rf_i].Dmap.unsqueeze(0), pred_smpl_verts)  # [B, 431, 3]
                ref_feature = self.maf_extractor[rf_i](pred_smpl_verts_ds)  # [B, 431 * n_feat] [b, 2155]

            smpl_output = self.regressor[rf_i](ref_feature, pred_pose, pred_shape, pred_cam, n_iter=1, J_regressor=J_regressor)
            out_list['smpl_out'].append(smpl_output)

        if self.cfg.train.backbone.aux_supv_on:
            iuv_out_dict = self.dp_head(s_feat)
            out_list['dp_out'].append(iuv_out_dict)

        return out_list, vis_feat_list

# if __name__ == '__main__':
#     # import numpy as np
#     # import os.path as osp
#     # import torch
#     #
#     # from d211227_pymaf_reimp.cfgs import ConfigPymaf
#     # from d211227_pymaf_reimp.nets import PyMAF
#     #
#     # cfg = ConfigPymaf()
#     # model = PyMAF(cfg.pymaf_model['BACKBONE'], cfg.res_model['DECONV_WITH_BIAS'], cfg.res_model['NUM_DECONV_LAYERS'],
#     #               cfg.res_model['NUM_DECONV_FILTERS'],
#     #               cfg.res_model['NUM_DECONV_KERNELS'], cfg.pymaf_model['MLP_DIM'], cfg.pymaf_model['N_ITER'],
#     #               cfg.pymaf_model['AUX_SUPV_ON'], cfg.BN_MOMENTUM,
#     #               cfg.SMPL_MODEL_DIR, cfg.H36M_TO_J14, cfg.LOSS['POINT_REGRESSION_WEIGHTS'], JOINT_MAP=cfg.JOINT_MAP,
#     #               JOINT_NAMES=cfg.JOINT_NAMES,
#     #               J24_TO_J19=cfg.J24_TO_J19, JOINT_REGRESSOR_TRAIN_EXTRA=cfg.JOINT_REGRESSOR_TRAIN_EXTRA,
#     #               device=cfg.device,
#     #               SMPL_MEAN_PARAMS_PATH=cfg.SMPL_MEAN_PARAMS_PATH, pretrained=True,
#     #               data_dir=cfg.preprocessed_data_dir).cuda()
#     #
#     # img = torch.randn(4, 3, 224, 224).cuda()
#     #
#     # outs = model(img)
#
#     pass