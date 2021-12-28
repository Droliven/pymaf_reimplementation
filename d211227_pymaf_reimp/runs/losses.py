#!/usr/bin/env python
# encoding: utf-8
'''
@project : pymaf_reimp
@file    : losses.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-12-27 20:04
'''
import torch
import torch.nn.functional as F

from ..utils.geometry import batch_rodrigues, perspective_projection, estimate_translation


def keypoint_loss(pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
    """ Compute 2D reprojection loss on the keypoints.
    The loss is weighted by the confidence.
    The available keypoints are different for each dataset.
    """
    conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
    conf[:, :25] *= openpose_weight
    conf[:, 25:] *= gt_weight
    loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
    return loss


def keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
    """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
    The loss is weighted by the confidence.
    """
    pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
    conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
    gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
    gt_keypoints_3d = gt_keypoints_3d[has_pose_3d]
    conf = conf[has_pose_3d]
    pred_keypoints_3d = pred_keypoints_3d[has_pose_3d]
    if len(gt_keypoints_3d) > 0:
        gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
        pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
        return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
    else:
        return torch.FloatTensor(1).fill_(0.).to(self.device)


def shape_loss(pred_vertices, gt_vertices, has_smpl):
    """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
    pred_vertices_with_shape = pred_vertices[has_smpl]
    gt_vertices_with_shape = gt_vertices[has_smpl]
    if len(gt_vertices_with_shape) > 0:
        return self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
    else:
        return torch.FloatTensor(1).fill_(0.).to(self.device)


def smpl_losses(pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl):
    pred_rotmat_valid = pred_rotmat[has_smpl]
    gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1, 3)).view(-1, 24, 3, 3)[has_smpl]
    pred_betas_valid = pred_betas[has_smpl]
    gt_betas_valid = gt_betas[has_smpl]
    if len(pred_rotmat_valid) > 0:
        loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
        loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
    else:
        loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
        loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
    return loss_regr_pose, loss_regr_betas


def body_uv_losses(u_pred, v_pred, index_pred, ann_pred, uvia_list, has_iuv=None, POINT_REGRESSION_WEIGHTS=0.5):
    batch_size = index_pred.size(0)
    device = index_pred.device

    Umap, Vmap, Imap, Annmap = uvia_list

    if has_iuv is not None:
        if torch.sum(has_iuv.float()) > 0:
            u_pred = u_pred[has_iuv] if u_pred is not None else u_pred
            v_pred = v_pred[has_iuv] if v_pred is not None else v_pred
            index_pred = index_pred[has_iuv] if index_pred is not None else index_pred
            ann_pred = ann_pred[has_iuv] if ann_pred is not None else ann_pred
            Umap, Vmap, Imap = Umap[has_iuv], Vmap[has_iuv], Imap[has_iuv]
            Annmap = Annmap[has_iuv] if Annmap is not None else Annmap
        else:
            return (
            torch.zeros(1).to(device), torch.zeros(1).to(device), torch.zeros(1).to(device), torch.zeros(1).to(device))

    Itarget = torch.argmax(Imap, dim=1)
    Itarget = Itarget.view(-1).to(torch.int64)

    index_pred = index_pred.permute([0, 2, 3, 1]).contiguous()
    index_pred = index_pred.view(-1, Imap.size(1))

    loss_IndexUV = F.cross_entropy(index_pred, Itarget)

    if POINT_REGRESSION_WEIGHTS > 0:
        loss_U = F.smooth_l1_loss(u_pred[Imap > 0], Umap[Imap > 0], reduction='sum') / batch_size
        loss_V = F.smooth_l1_loss(v_pred[Imap > 0], Vmap[Imap > 0], reduction='sum') / batch_size

        loss_U *= POINT_REGRESSION_WEIGHTS
        loss_V *= POINT_REGRESSION_WEIGHTS
    else:
        loss_U, loss_V = torch.zeros(1).to(device), torch.zeros(1).to(device)

    if ann_pred is None:
        loss_segAnn = None
    else:
        Anntarget = torch.argmax(Annmap, dim=1)
        Anntarget = Anntarget.view(-1).to(torch.int64)
        ann_pred = ann_pred.permute([0, 2, 3, 1]).contiguous()
        ann_pred = ann_pred.view(-1, Annmap.size(1))
        loss_segAnn = F.cross_entropy(ann_pred, Anntarget)

    return loss_U, loss_V, loss_IndexUV, loss_segAnn
