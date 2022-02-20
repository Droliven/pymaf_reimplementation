#!/usr/bin/env python
# encoding: utf-8
'''
@project : pymaf_reimp
@file    : run_pymafreimp.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2022-01-21 10:57
'''

from ..datas import MixedTrainDS, BaseDS, FitsD
from ..nets import PyMAF, SMPL
from ..cfgs import get_cfg_pymafreimp, BaseDict
from .losses import smpl_losses, body_uv_losses, keypoint_loss, keypoint_3d_loss
from ..utils.pose_utils import reconstruction_error
from ..utils.geometry import projection, estimate_translation, rotation_matrix_to_angle_axis
# from ..utils.renderer import SpinPyRenderer, IUV_Renderer
# from ..utils.draw_skeleton_in_img import draw_pic_gt_pred_2d
from ..utils.iuvmap import iuv_img2map, iuv_map2img

import numpy as np
import time
from torch.optim import Adam
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import os.path as osp
from tensorboardX import SummaryWriter
from tqdm import tqdm
from pprint import pprint
import json
import torch.nn.functional as F
from skimage.transform import resize
import datetime


# from torchgeometry import rotation_matrix_to_angle_axis


class RunPymafReimp():
    def __init__(self, exp_name="", is_debug=False, args=None):
        super(RunPymafReimp, self).__init__()

        self.is_debug = is_debug

        # 参数
        self.epoch_count = 0
        self.step_count = 0
        self.best_performance = float('inf')

        cfg_json_dict = get_cfg_pymafreimp(exp_name=exp_name, is_debug=is_debug)
        self.cfg = BaseDict(cfg_json_dict)

        print("\n================== Configs =================")
        print(json.dumps(cfg_json_dict, indent=4, ensure_ascii=False, separators=(", ", ": ")))
        print("==========================================\n")

        save_dict = {"args": args.__dict__, "cfgs": self.cfg}
        save_json = json.dumps(save_dict)

        with open(os.path.join(self.cfg.run.output_dir, "config.json"), 'w', encoding='utf-8') as f:
            f.write(save_json)

        # log
        self.summary = SummaryWriter(self.cfg.run.output_dir)

        # 模型
        # PyMAF model
        self.model = PyMAF(self.cfg, pretrained=True)
        self.smpl = self.model.regressor[0].smpl
        # Load SMPL model
        self.smpl_neutral = SMPL(self.cfg, self.cfg.run.smpl_model_path, batch_size=self.cfg.train.batch_size, create_transl=False)
        self.smpl_male = SMPL(self.cfg, self.cfg.run.smpl_model_path, batch_size=self.cfg.train.batch_size, gender='male', create_transl=False)
        self.smpl_female = SMPL(self.cfg, self.cfg.run.smpl_model_path, batch_size=self.cfg.train.batch_size, gender='female',
                                create_transl=False)

        if self.cfg.run.device != "cpu":
            self.model.cuda(self.cfg.run.device)
            self.smpl.cuda(self.cfg.run.device)
            self.smpl_neutral.cuda(self.cfg.run.device)
            self.smpl_male.cuda(self.cfg.run.device)
            self.smpl_female.cuda(self.cfg.run.device)

        print(">>> total params of {}: {:.6f}M\n".format(exp_name,
                                                         sum(p.numel() for p in self.model.parameters()) / 1000000.0))

        self.optimizer = Adam(self.model.parameters(), lr=self.cfg.train.lr, weight_decay=0)

        # Per-vertex loss on the shape
        self.criterion_shape = nn.L1Loss().to(self.cfg.run.device)
        # Keypoint (2D and 3D) loss
        # No reduction because confidence weighting needs to be applied
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.cfg.run.device)
        # Loss for SMPL parameter regression
        self.criterion_regr = nn.MSELoss().to(self.cfg.run.device)

        # 数据
        if args.is_single_dataset:
            train_ds = BaseDS(is_debug, self.cfg, self.cfg.dataset.train_list[0], ignore_3d=False, use_augmentation=True, is_train=True)
        else:
            train_ds = MixedTrainDS(is_debug, self.cfg, ignore_3d=False, use_augmentation=True, is_train=True)

        # Load dictionary of fits
        self.fits_dict = FitsD(is_debug, self.cfg, args.is_single_dataset, train_ds)
        self.train_data_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.run.num_works,
            # pin_memory=True,
            shuffle=True,
        )
        valid_ds = BaseDS(is_debug=self.is_debug, cfg=self.cfg, data_cfg=self.cfg.dataset.test_list[0], ignore_3d=False, use_augmentation=True, is_train=False) # test_h36mp1
        # valid_ds = BaseDS(is_debug=self.is_debug, cfg=self.cfg, data_cfg=self.cfg.dataset.test_list[1], ignore_3d=False, use_augmentation=True, is_train=False) # test_h36mp2
        # valid_ds = BaseDS(is_debug=self.is_debug, cfg=self.cfg, data_cfg=self.cfg.dataset.test_list[2], ignore_3d=False, use_augmentation=True, is_train=False) # test_h36mp2mosh
        # valid_ds = BaseDS(is_debug=self.is_debug, cfg=self.cfg, data_cfg=self.cfg.dataset.test_list[5], ignore_3d=False, use_augmentation=True, is_train=False) # 3dpw
        self.valid_loader = DataLoader(
            dataset=valid_ds,
            batch_size=self.cfg.train.batch_size,
            # batch_size=32,
            shuffle=False,
            num_workers=self.cfg.run.num_works,
            # pin_memory=True,
        )

        self.J_regressor = torch.from_numpy(np.load(self.cfg.run.J_regressor_h36m_path)).float()
        self.joint_mapper_h36m = self.cfg.constants.h36m_to_j17 if valid_ds.dataset == 'test_mpiinf3dhp' else self.cfg.constants.h36m_to_j14
        self.joint_mapper_gt = self.cfg.constants.j24_to_j17 if valid_ds.dataset == 'test_mpiinf3dhp' else self.cfg.constants.j24_to_j14

        # # Create renderer
        # try:
        #     # self.renderer = OpenDRenderer()
        #     self.renderer = SpinPyRenderer(focal_length=self.cfg.constants.focal_length,
        #                              img_res=self.cfg.constants.img_size[0], faces=self.smpl.faces)
        # except:
        #     print('No renderer for visualization.')
        #     self.renderer = None
        #
        # if self.cfg.train.backbone.aux_supv_on:
        #     try:
        #         self.iuv_maker = IUV_Renderer(focal_length=self.cfg.constants.focal_length, orig_size=self.cfg.consatants.img_size[0], output_size=self.cfg.train.backbone.dp_heatmap_size, UV_data_path=self.cfg.run.uv_data_path)
        #     except:
        #         print('No IUV_Renderer.')
        #         self.iuv_maker = None

        self.renderer = None
        self.iuv_maker = None

    def save_checkpoint(self, epoch, is_best=False, performance=-1):
        """Save checkpoint."""
        checkpoint_filename = os.path.join(self.cfg.run.output_dir, "models",
                                           f'{self.cfg.run.exp_name}_epoch_{epoch:08d}_performance_{performance:.6f}.pth')

        checkpoint = {}
        checkpoint["model"] = self.model.state_dict()
        checkpoint["optimizer"] = self.optimizer.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint["performance"] = performance

        if checkpoint_filename is not None:
            torch.save(checkpoint, checkpoint_filename)
        if is_best:  # save the best
            checkpoint_filename = os.path.join(self.cfg.run.output_dir, "models", f'{self.cfg.run.exp_name}_best.pth')
            torch.save(checkpoint, checkpoint_filename)

    def load_checkpoint(self, checkpoint_file=None):
        """Load a checkpoint."""

        checkpoint = torch.load(checkpoint_file)

        if "model" in checkpoint:
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint["model"].items() if k in model_dict.keys() and model_dict[k].shape == v.shape}
            # for i, k in enumerate(model_dict.keys()):
            #     if model_dict[k].shape != pretrained_dict[k].shape:
            #         print(f"{i}, {k}: {model_dict[k].shape} || {pretrained_dict[k].shape}")
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)

        # self.optimizer.load_state_dict(checkpoint["optimizer"])
        # self.epoch_count = checkpoint['epoch']

        # print(
        #     f"load from {checkpoint_file}, epoch: {checkpoint['epoch']}, performance: {checkpoint['performance']}")
        print(f"load from {checkpoint_file}")

    def train_batch(self, input_batch):
        self.model.train()
        """Training process."""
        # >>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<
        # >>>>>>>>>>>>>> train on batch <<<<<<<<<<<<<<<<
        # >>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<
        # Get data from the batch
        images = input_batch['img']  # input image
        gt_keypoints_2d = input_batch['keypoints']  # 2D keypoints
        gt_pose = input_batch['pose']  # SMPL pose parameters
        gt_betas = input_batch['betas']  # SMPL beta parameters
        gt_joints = input_batch['pose_3d']  # 3D pose
        has_smpl = input_batch['has_smpl'].to(torch.bool)  # flag that indicates whether SMPL parameters are valid
        has_pose_3d = input_batch['has_pose_3d'].to(torch.bool)  # flag that indicates whether 3D pose is valid
        is_flipped = input_batch['is_flipped']  # flag that indicates whether image was flipped during data augmentation
        rot_angle = input_batch['rot_angle']  # rotation angle used for data augmentation
        dataset_name = input_batch['dataset_name']  # name of the dataset the image comes from
        indices = input_batch['sample_index']  # index of example inside its dataset
        batch_size = images.shape[0]

        # Get current best fits from the dictionary
        opt_pose, opt_betas = self.fits_dict[(dataset_name, indices.cpu(), rot_angle.cpu(), is_flipped.cpu())]
        opt_pose = opt_pose.to(self.cfg.run.device)
        opt_betas = opt_betas.to(self.cfg.run.device)
        # Replace extreme betas with zero betas
        opt_betas[(opt_betas.abs() > 3).any(dim=-1)] = 0.
        # Replace the optimized parameters with the ground truth parameters, if available
        opt_pose[has_smpl, :] = gt_pose[has_smpl, :]
        opt_betas[has_smpl, :] = gt_betas[has_smpl, :]

        opt_output = self.smpl(betas=opt_betas, body_pose=opt_pose[:, 3:], global_orient=opt_pose[:, :3])
        opt_vertices = opt_output.vertices  # [b, 6890, 3]
        opt_joints = opt_output.joints  # [b, 49, 3]
        input_batch['verts'] = opt_vertices
        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, :-1] = 0.5 * self.cfg.constants.img_size[0] * (gt_keypoints_2d_orig[:, :, :-1] + 1)

        # Estimate camera translation given the model joints and 2D keypoints
        # by minimizing a weighted least squares loss

        opt_cam_t = estimate_translation(opt_joints, gt_keypoints_2d_orig, focal_length=self.cfg.constants.focal_length,
                                         img_size=self.cfg.constants.img_size[0])
        # get fitted smpl parameters as pseudo ground truth, todo 这玩意儿代表啥
        valid_fit = self.fits_dict.get_vaild_state(dataset_name, indices.cpu()).to(torch.bool).to(
            self.cfg.run.device)  # [b]

        try:
            valid_fit = valid_fit | has_smpl
        except RuntimeError:
            valid_fit = (valid_fit.byte() | has_smpl.byte()).to(torch.bool)

        # Render Dense Correspondences
        if self.cfg.train.backbone.aux_supv_on:
            gt_cam_t_nr = opt_cam_t.detach().clone()
            gt_camera = torch.zeros(gt_cam_t_nr.shape).to(gt_cam_t_nr.device)
            gt_camera[:, 1:] = gt_cam_t_nr[:, :2]  # [tx, ty, tz] -> [s, tx, ty]
            gt_camera[:, 0] = (2. * self.cfg.constants.focal_length / self.cfg.constants.img_size[0]) / gt_cam_t_nr[:, 2]
            iuv_image_gt = torch.zeros(
                (batch_size, 3, self.cfg.train.backbone.dp_heatmap_size, self.cfg.train.backbone.dp_heatmap_size)).to(
                self.cfg.run.device)  # [b, 3, 56, 56]
            if torch.sum(valid_fit.float()) > 0 and self.iuv_maker is not None:
                iuv_image_gt[valid_fit] = self.iuv_maker.verts2iuvimg(opt_vertices[valid_fit],
                                                                      cam=gt_camera[valid_fit])  # [B, 3, 56, 56]
            input_batch['iuv_image_gt'] = iuv_image_gt  # # [b, 3, 56, 56]
            # todo iuv_img2map 是什么
            uvia_list = iuv_img2map(iuv_image_gt)  # [b, 25, 56, 56] * 4

        # >>>>>>>>>>>>>>>>>>>>>> Feed images in the network to predict camera and SMPL parameters <<<<<<<<<<<<<<<<<<<<<<
        preds_dict, _ = self.model(images)
        # {"smpl_out": {'theta': [b, 85], 'verts': [b, 6890, 3], 'rotmat': [b, 24, 3, 3], 'pred_pose': [b, 144]} * 4,
        #  "dp_out": {'predict_uv_index': [b, 25, 56, 56], 'predict_ann_index': [b, 15, 56, 56], 'predict_u': [b, 25, 56, 56], 'predict_v': [b, 25, 56, 56]} * 1}
        # [b, 2048, 7, 7], [b, 256, 14, 14], [b, 256, 28, 28], [b, 256, 56, 56]
        loss_dict = {}
        if self.cfg.train.backbone.aux_supv_on:
            dp_out = preds_dict['dp_out']
            for i in range(len(dp_out)):
                r_i = i - len(dp_out)

                u_pred, v_pred, index_pred, ann_pred = dp_out[r_i]['predict_u'], dp_out[r_i]['predict_v'], \
                                                       dp_out[r_i]['predict_uv_index'], dp_out[r_i][
                                                           'predict_ann_index']
                if index_pred.shape[-1] == iuv_image_gt.shape[-1]:
                    uvia_list_i = uvia_list
                else:
                    iuv_image_gt_i = F.interpolate(iuv_image_gt, u_pred.shape[-1], mode='nearest')
                    uvia_list_i = iuv_img2map(iuv_image_gt_i)

                loss_U, loss_V, loss_IndexUV, loss_segAnn = body_uv_losses(u_pred, v_pred, index_pred,
                                                                           ann_pred,
                                                                           uvia_list_i, valid_fit)
                loss_dict[f'loss/U{r_i}'] = loss_U
                loss_dict[f'loss/V{r_i}'] = loss_V
                loss_dict[f'loss/IndexUV{r_i}'] = loss_IndexUV
                loss_dict[f'loss/segAnn{r_i}'] = loss_segAnn

        len_loop = len(preds_dict['smpl_out'])
        for l_i in range(len_loop):

            if l_i == 0:
                # initial parameters (mean poses)
                continue
            pred_rotmat = preds_dict['smpl_out'][l_i]['rotmat']
            pred_betas = preds_dict['smpl_out'][l_i]['theta'][:, 3:13]
            pred_camera = preds_dict['smpl_out'][l_i]['theta'][:, :3]  # 这里出来的是 [s, tx, ty]

            pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:, 1:], global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
            pred_joints = pred_output.joints  # 世界坐标系下的 3D
            pred_keypoints_2d = projection(pred_joints, pred_camera, retain_z=False)
            # Compute loss on SMPL parameters
            loss_regr_pose, loss_regr_betas = smpl_losses(self.criterion_regr, pred_rotmat, pred_betas, opt_pose, opt_betas, valid_fit)
            loss_dict['loss/smpl_pose_{}'.format(l_i)] = loss_regr_pose * self.cfg.train.pose_loss_weight
            loss_dict['loss/smpl_betas_{}'.format(l_i)] = loss_regr_betas * self.cfg.train.beta_loss_weight

            loss_keypoints = keypoint_loss(self.criterion_keypoints, pred_keypoints_2d, gt_keypoints_2d, self.cfg.train.openpose_train_weight,
                                               self.cfg.train.gt_train_weight) * self.cfg.train.keypoint_loss_weight
            loss_dict['loss/keypoints2d_{}'.format(l_i)] = loss_keypoints

            # Compute 3D keypoint loss
            loss_keypoints_3d = keypoint_3d_loss(self.criterion_keypoints, pred_joints, gt_joints, has_pose_3d) * self.cfg.train.keypoint_loss_weight
            loss_dict['loss/keypoints3d_{}'.format(l_i)] = loss_keypoints_3d
            # Camera
            # force the network to predict positive depth values
            loss_cam = ((torch.exp(-pred_camera[:, 0] * 10)) ** 2).mean()  # 用的是 [s, tx, ty]
            loss_dict['loss/cam_{}'.format(l_i)] = loss_cam

        for key in loss_dict:
            if len(loss_dict[key].shape) > 0:
                loss_dict[key] = loss_dict[key][0]

        # Compute total loss
        loss = torch.stack(list(loss_dict.values())).sum()

        # Do backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # todo 要理清楚 predxxx 与 opt xxx 的关系
        # Pack output arguments for tensorboard logging
        output = {'pred_vertices': preds_dict["smpl_out"][-1]['verts'].detach(),
                  'opt_vertices': opt_vertices.detach(),
                  'pred_cam_t': torch.stack([pred_camera[:, 1],
                                          pred_camera[:, 2],
                                          2 * self.cfg.constants.focal_length / (self.cfg.constants.img_size[0] * pred_camera[:, 0] + 1e-9)],
                                         dim=-1).detach(),
                  'opt_cam_t': opt_cam_t.detach(),
                  'pred_keypoint_2d_origin': 0.5 * self.cfg.constants.img_size[0] * (pred_keypoints_2d[:, :, :-1].detach() + 1),
                  'gt_keypoint_2d_origin': gt_keypoints_2d_orig.detach()}
        loss_dict['loss'] = loss.detach().item()

        return output, loss_dict

    def log_train(self, input_batch, output, losses):
        images = input_batch['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)

        pred_vertices = output['pred_vertices']
        opt_vertices = output['opt_vertices']
        pred_cam_t = output['pred_cam_t']
        opt_cam_t = output['opt_cam_t']
        pred_keypoint_2d_origin = output['pred_keypoint_2d_origin']
        gt_keypoint_2d_origin = output['gt_keypoint_2d_origin']

        vis_n = min(images.shape[0], 4)

        images = images[:vis_n]
        pred_vertices = pred_vertices[:vis_n]
        opt_vertices = opt_vertices[:vis_n]
        pred_cam_t = pred_cam_t[:vis_n]
        opt_cam_t = opt_cam_t[:vis_n]
        # pred_keypoint_2d_origin = pred_keypoint_2d_origin[:vis_n]
        # gt_keypoint_2d_origin = gt_keypoint_2d_origin[:vis_n, :, :2]

        # skeleton_img = draw_pic_gt_pred_2d(images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8) * 255, gt_keypoint_2d_origin.cpu().data.numpy(), pred_keypoint_2d_origin.cpu().data.numpy())
        if self.renderer is not None:
            images_pred = self.renderer.visualize_tb(pred_vertices, pred_cam_t, images)
            images_opt = self.renderer.visualize_tb(opt_vertices, opt_cam_t, images)
            self.summary.add_image('train/pred_shape', images_pred, self.step_count)
            self.summary.add_image('train/opt_shape', images_opt, self.step_count)

        for loss_name, val in losses.items():
            self.summary.add_scalar(loss_name, val, self.step_count)

    def test(self):
        self.model.eval()
        dataset_name = self.valid_loader.dataset.dataset
        assert dataset_name in ['test_h36mp1', 'test_h36mp2', 'test_h36mp2mosh', 'test_mpiinf3dhp', 'test_3dpw']

        # Pose metrics
        # MPJPE and Reconstruction error for the non-parametric and parametric shapes
        mpjpe = np.zeros(len(self.valid_loader.dataset))
        pampjpe = np.zeros(len(self.valid_loader.dataset))
        pve = np.zeros(len(self.valid_loader.dataset))

        # Iterate over the entire dataset
        for step, batch in enumerate(
                tqdm(self.valid_loader, desc=f">>> Test {dataset_name}", total=len(self.valid_loader))):
            # # De-normalize 2D keypoints from [-1,1] to pixel space
            # gt_keypoints_2d_orig = batch['keypoints'].to(self.cfg.run.device)
            # gt_keypoints_2d_orig[:, :, :-1] = 0.5 * self.cfg.constants.img_size[0] * (gt_keypoints_2d_orig[:, :, :-1] + 1)
            # Get ground truth annotations from the batch
            gt_pose = batch['pose'].to(self.cfg.run.device)
            gt_betas = batch['betas'].to(self.cfg.run.device)
            gt_smpl_outs = self.smpl_neutral(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3])
            gt_vertices = gt_smpl_outs.vertices
            gt_keypoints_3d_49 = gt_smpl_outs.joints
            images = batch['img'].to(self.cfg.run.device)
            gender = batch['gender'].to(self.cfg.run.device)

            curr_batch_size = images.shape[0]
            with torch.no_grad():
                preds_dict, _ = self.model(images)
                pred_rotmat = preds_dict['smpl_out'][-1]['rotmat'].contiguous().view(-1, 24, 3, 3)
                pred_betas = preds_dict['smpl_out'][-1]['theta'][:, 3:13].contiguous()
                pred_camera = preds_dict['smpl_out'][-1]['theta'][:, :3].contiguous()
                # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
                # This camera translation can be used in a full perspective projection
                pred_cam_t = torch.stack([pred_camera[:, 1],
                                          pred_camera[:, 2],
                                          2 * self.cfg.constants.focal_length / (self.cfg.constants.img_size[0] * pred_camera[:, 0] + 1e-9)],
                                         dim=-1)

                pred_output = self.smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                        global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
                pred_vertices = pred_output.vertices  # todo 这里为什么又去构造一次，而不直接去用输出的 vertices
                pred_joints = pred_output.joints  # 世界坐标系下的 3D
                pred_keypoints_2d = projection(pred_joints, pred_camera, retain_z=False)

            # 3D pose evaluation
            # Regressor broadcasting
            J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(self.cfg.run.device)
            # Get 14 ground truth joints
            if dataset_name == 'test_h36mp1' or dataset_name == 'test_h36mp2' or dataset_name == 'test_h36mp2mosh' or dataset_name == 'test_mpiinf3dhp':
                gt_keypoints_3d = batch['pose_3d'].to(self.cfg.run.device)
                gt_keypoints_3d = gt_keypoints_3d[:, self.joint_mapper_gt, :-1]

            # For 3DPW get the 14 common joints from the rendered shape
            elif dataset_name == "test_3dpw":
                gt_vertices = self.smpl_male(global_orient=gt_pose[:, :3], body_pose=gt_pose[:, 3:],
                                             betas=gt_betas).vertices
                gt_vertices_female = self.smpl_female(global_orient=gt_pose[:, :3], body_pose=gt_pose[:, 3:],
                                                      betas=gt_betas).vertices
                gt_vertices[gender == 1, :, :] = gt_vertices_female[gender == 1, :, :]
                gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
                gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
                gt_keypoints_3d = gt_keypoints_3d[:, self.joint_mapper_h36m, :]
                gt_keypoints_3d = gt_keypoints_3d - gt_pelvis

            per_vertex_error = torch.sqrt(((pred_vertices - gt_vertices) ** 2).sum(dim=-1)).mean(
                    dim=-1).cpu().numpy()

            pve[step * self.valid_loader.batch_size:step * self.valid_loader.batch_size + curr_batch_size] = per_vertex_error
            # Get 14 predicted joints from the mesh
            pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
            pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, self.joint_mapper_h36m, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis

            # Absolute error (MPJPE)
            error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            mpjpe[
            step * self.valid_loader.batch_size:step * self.valid_loader.batch_size + curr_batch_size] = error

            # PAMPJPE
            r_error, _ = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(), reduction=None)
            pampjpe[step * self.valid_loader.batch_size:step * self.valid_loader.batch_size + curr_batch_size] = r_error

            # print(f'Test {self.valid_loader.dataset.dataset}, step {step} || MPJPE: {1000 * error.mean()}, PAMPJPE: {1000 * r_error.mean()}, PVE: {1000 * per_vertex_error.mean()}')

            # >>>>> 插入可视化 mesh 的部分
            # if step == 0:
            #     vis_n = min(curr_batch_size, 4)
            #     images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
            #     images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)
            #
            #     gt_cam_t = estimate_translation(gt_keypoints_3d_49, gt_keypoints_2d_orig,
            #                                     focal_length=self.cfg.constants.focal_length,
            #                                     img_size=self.cfg.constants.img_size[0])
            #
            #     images = images[:vis_n]
            #     pred_cam_t = pred_cam_t[:vis_n]
            #     gt_cam_t = gt_cam_t[:vis_n]
            #     gt_vertices = gt_vertices[:vis_n]
            #     pred_vertices = pred_vertices[:vis_n]
            #     if not self.renderer is None:
            #         images_gt = self.renderer.visualize_tb(gt_vertices, gt_cam_t, images)
            #         images_pred = self.renderer.visualize_tb(pred_vertices, pred_cam_t, images)
            #         self.summary.add_image('test/gt_mesh', images_gt, self.step_count)
            #         self.summary.add_image('test/pred_mesh', images_pred, self.step_count)

        self.summary.add_scalar(f"test/pve", 1000 * pve.mean(), self.step_count)
        self.summary.add_scalar(f"test/mpjpe", 1000 * mpjpe.mean(), self.step_count)
        self.summary.add_scalar(f"test/pampjpe", 1000 * pampjpe.mean(), self.step_count)

        return 1000 * mpjpe.mean(), 1000 * pampjpe.mean(), 1000 * pve.mean()

    def run(self):
        # for epoch in tqdm(range(self.epoch_count, self.cfg.epoch_num), total=self.cfg.epoch_num, initial=self.epoch_count):
        for epoch in range(self.epoch_count, self.cfg.train.num_epochs):
            self.epoch_count = epoch
            # Iterate over all batches in an epoch
            for step, input_batch in enumerate(
                    tqdm(self.train_data_loader, desc=f">>> epoch {epoch:>5d}", total=len(self.train_data_loader))):
                # ['img', 'pose', 'shape', 'imgname', 'smpl_2dkps', 'pose_3d', 'keypoints', 'has_smpl', 'has_pose_3d', 'scale',
                # 'center', 'orig_shape', 'is_flipped', 'rot_angle', 'gender', 'sample_index', 'dataset_name', 'maskname', 'partname']
                input_batch = {k: v.to(self.cfg.run.device) if isinstance(v, torch.Tensor) else v for k, v in
                               input_batch.items()}

                output, losses = self.train_batch(input_batch)

                self.step_count += 1

                # Tensorboard logging every summary_steps steps
                if self.step_count % self.cfg.train.log_steps == 0:
                    self.log_train(input_batch, output, losses)

                # Run validation every test_steps steps
                if self.step_count % self.cfg.train.test_steps == 0:
                    mpjpe, pampjpe, pve = self.test()
                    print(
                        f'Test {self.valid_loader.dataset.dataset} || MPJPE: {mpjpe.mean()}, PAMPJPE: {pampjpe.mean()}, PVE: {pve.mean()}')

                # Save checkpoint every checkpoint_steps steps
                if self.step_count % self.cfg.train.save_steps == 0:
                    self.save_checkpoint(self.epoch_count)

            print()

        self.fits_dict.save()







