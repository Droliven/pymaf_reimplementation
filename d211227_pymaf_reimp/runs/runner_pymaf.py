#!/usr/bin/env python
# encoding: utf-8
'''
@project : pymaf_reimp
@file    : runner_pymaf.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-12-27 15:55
'''
from ..datas import MixedDataset, BaseDataset
from ..nets import PyMAF
from ..cfgs import ConfigPymaf
from .losses import smpl_losses, body_uv_losses, shape_loss, keypoint_loss, keypoint_3d_loss
from .fits_dit import FitsDict
from ..utils.geometry import batch_rodrigues, perspective_projection, estimate_translation
from ..utils.iuvmap import iuv_img2map, iuv_map2img
from ..utils.renderer import OpenDRenderer, IUV_Renderer
from ..utils.pose_utils import compute_similarity_transform_batch

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


class RunnerPymaf():
    def __init__(self, exp_name="", is_debug=False, args=None):
        super(RunnerPymaf, self).__init__()

        self.is_debug = is_debug

        # 参数
        self.epoch_count = 0
        self.step_count = 0
        self.decay_steps_ind = 1
        self.decay_epochs_ind = 1

        self.best_performance = float('inf')
        self.checkpoint_batch_idx = 0

        self.cfg = ConfigPymaf(exp_name=exp_name)

        print("\n================== Configs =================")
        pprint(vars(self.cfg), indent=4)
        print("==========================================\n")

        save_dict = {"args": args.__dict__, "cfgs": self.cfg.__dict__}
        save_json = json.dumps(save_dict)

        with open(os.path.join(self.cfg.output_dir, "config.json"), 'w', encoding='utf-8') as f:
            f.write(save_json)

        # log
        self.summary = SummaryWriter(self.cfg.output_dir)

        # 模型
        # PyMAF model
        self.model = PyMAF(self.cfg.pymaf_model['BACKBONE'], self.cfg.res_model['DECONV_WITH_BIAS'], self.cfg.res_model['NUM_DECONV_LAYERS'], self.cfg.res_model['NUM_DECONV_FILTERS'], self.cfg.res_model['NUM_DECONV_KERNELS'], self.cfg.pymaf_model['MLP_DIM'], self.cfg.pymaf_model['N_ITER'], self.cfg.pymaf_model['AUX_SUPV_ON'], self.cfg.BN_MOMENTUM, self.cfg.SMPL_MODEL_DIR, self.cfg.H36M_TO_J14, self.cfg.LOSS['POINT_REGRESSION_WEIGHTS'],
                            JOINT_MAP=self.cfg.JOINT_MAP, JOINT_NAMES=self.cfg.JOINT_NAMES, J24_TO_J19=self.cfg.J24_TO_J19, JOINT_REGRESSOR_TRAIN_EXTRA=self.cfg.JOINT_REGRESSOR_TRAIN_EXTRA,
                            device=self.cfg.device, SMPL_MEAN_PARAMS_PATH=self.cfg.SMPL_MEAN_PARAMS_PATH, pretrained=True, data_dir=self.cfg.preprocessed_data_dir)
        self.smpl = self.model.regressor[0].smpl

        if self.cfg.device != "cpu":
            self.model.cuda(self.cfg.device)

        print(">>> total params of {}: {:.6f}M\n".format(exp_name, sum(p.numel() for p in self.model.parameters()) / 1000000.0))

        # Per-vertex loss on the shape
        self.criterion_shape = nn.L1Loss().to(self.cfg.device)
        # Keypoint (2D and 3D) loss
        # No reduction because confidence weighting needs to be applied
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.cfg.device)
        # Loss for SMPL parameter regression
        self.criterion_regr = nn.MSELoss().to(self.cfg.device)
        self.focal_length = self.cfg.FOCAL_LENGTH

        self.optimizer = Adam(self.model.parameters(), lr=self.cfg.SOLVER['BASE_LR'], weight_decay=0)
        self.optimizers_dict = {'optimizer': self.optimizer}

        # 数据
        if args.is_single_dataset:
            self.train_ds = BaseDataset(eval_pve=self.cfg.eval_pve, noise_factor=self.cfg.noise_factor, rot_factor=self.cfg.rot_factor, scale_factor=self.cfg.scale_factor, dataset=self.cfg.single_dataname, ignore_3d=False, use_augmentation=True, is_train=True, DATASET_FOLDERS=self.cfg.ORIGIN_IMGS_DATASET_FOLDERS, DATASET_FILES=self.cfg.PREPROCESSED_DATASET_FILES, JOINT_MAP=self.cfg.JOINT_MAP, JOINT_NAMES=self.cfg.JOINT_NAMES, J24_TO_J19=self.cfg.J24_TO_J19, JOINT_REGRESSOR_TRAIN_EXTRA=self.cfg.JOINT_REGRESSOR_TRAIN_EXTRA, SMPL_MODEL_DIR=self.cfg.SMPL_MODEL_DIR, IMG_NORM_MEAN=self.cfg.IMG_NORM_MEAN, IMG_NORM_STD=self.cfg.IMG_NORM_STD, TRAIN_BATCH_SIZE=self.cfg.TRAIN_BATCHSIZE, IMG_RES=self.cfg.IMG_RES, SMPL_JOINTS_FLIP_PERM=self.cfg.SMPL_JOINTS_FLIP_PERM)
        else:
            self.train_ds = MixedDataset(eval_pve=self.cfg.eval_pve, noise_factor=self.cfg.noise_factor, rot_factor=self.cfg.rot_factor, scale_factor=self.cfg.scale_factor, ignore_3d=False, use_augmentation=True, is_train=True, DATASET_FOLDERS=self.cfg.ORIGIN_IMGS_DATASET_FOLDERS, DATASET_FILES=self.cfg.PREPROCESSED_DATASET_FILES, JOINT_MAP=self.cfg.JOINT_MAP, JOINT_NAMES=self.cfg.JOINT_NAMES, J24_TO_J19=self.cfg.J24_TO_J19, JOINT_REGRESSOR_TRAIN_EXTRA=self.cfg.JOINT_REGRESSOR_TRAIN_EXTRA, SMPL_MODEL_DIR=self.cfg.SMPL_MODEL_DIR, IMG_NORM_MEAN=self.cfg.IMG_NORM_MEAN, IMG_NORM_STD=self.cfg.IMG_NORM_STD, TRAIN_BATCH_SIZE=self.cfg.TRAIN_BATCHSIZE, IMG_RES=self.cfg.IMG_RES, SMPL_JOINTS_FLIP_PERM=self.cfg.SMPL_JOINTS_FLIP_PERM)

        self.valid_ds = BaseDataset(eval_pve=self.cfg.eval_pve, noise_factor=self.cfg.noise_factor, rot_factor=self.cfg.rot_factor, scale_factor=self.cfg.scale_factor, dataset=self.cfg.eval_dataset, ignore_3d=False, use_augmentation=True, is_train=False, DATASET_FOLDERS=self.cfg.ORIGIN_IMGS_DATASET_FOLDERS, DATASET_FILES=self.cfg.PREPROCESSED_DATASET_FILES, JOINT_MAP=self.cfg.JOINT_MAP, JOINT_NAMES=self.cfg.JOINT_NAMES, J24_TO_J19=self.cfg.J24_TO_J19, JOINT_REGRESSOR_TRAIN_EXTRA=self.cfg.JOINT_REGRESSOR_TRAIN_EXTRA, SMPL_MODEL_DIR=self.cfg.SMPL_MODEL_DIR, IMG_NORM_MEAN=self.cfg.IMG_NORM_MEAN, IMG_NORM_STD=self.cfg.IMG_NORM_STD, TRAIN_BATCH_SIZE=self.cfg.TRAIN_BATCHSIZE, IMG_RES=self.cfg.IMG_RES, SMPL_JOINTS_FLIP_PERM=self.cfg.SMPL_JOINTS_FLIP_PERM)

        self.train_data_loader = DataLoader(
            self.train_ds,
            batch_size=self.cfg.TRAIN_BATCHSIZE,
            num_workers=self.cfg.num_works,
            pin_memory=True,
            shuffle=True,
        )

        self.valid_loader = DataLoader(
            dataset=self.valid_ds,
            batch_size=self.cfg.TEST_BATCHSIZE,
            shuffle=False,
            num_workers=self.cfg.num_works,
            pin_memory=True,
        )

        # Load dictionary of fits
        self.fits_dict = FitsDict(is_single_dataset=args.is_single_dataset, output_dir=self.cfg.output_dir, train_dataset=self.train_ds, SMPL_POSE_FLIP_PERM=self.cfg.SMPL_POSE_FLIP_PERM, FINAL_FITS_DIR=self.cfg.FINAL_FITS_DIR)
        self.evaluation_accumulators = dict.fromkeys(['pred_j3d', 'target_j3d', 'target_theta', 'pred_verts', 'target_verts'])

        # Create renderer
        try:
            self.renderer = OpenDRenderer()
        except:
            print('No renderer for visualization.')
            self.renderer = None

        if self.cfg.pymaf_model['AUX_SUPV_ON']:
            self.iuv_maker = IUV_Renderer(output_size=self.cfg.pymaf_model['DP_HEATMAP_SIZE'], UV_data_path=self.cfg.UV_data_path)


    def finalize(self):
        self.fits_dict.save()

    def save_checkpoint(self, model, optimizer, epoch, batch_idx, batch_size, total_step_count, is_best=False, save_by_step=False, interval=5):
        """Save checkpoint."""
        timestamp = datetime.datetime.now()
        checkpoint_filename = os.path.abspath(os.path.join(self.cfg.output_dir, "models", f'model_epoch_{epoch:08d}.pt'))

        checkpoint = {}
        model_dict = model.state_dict()
        for k in list(model_dict.keys()):
            if k.startswith('iuv2smpl.smpl.'):
                del model_dict[k]
        checkpoint["model"] = model_dict
        checkpoint["optimizer"] = optimizer.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['batch_idx'] = batch_idx
        checkpoint['batch_size'] = batch_size
        checkpoint['total_step_count'] = total_step_count
        print(timestamp, 'Epoch:', epoch, 'Iteration:', batch_idx)

        if checkpoint_filename is not None:
            torch.save(checkpoint, checkpoint_filename)
            print('Saving checkpoint file [' + checkpoint_filename + ']')
        if is_best:  # save the best
            checkpoint_filename = os.path.abspath(os.path.join(self.cfg.output_dir, "models", 'model_best.pt'))
            torch.save(checkpoint, checkpoint_filename)
            print(timestamp, 'Epoch:', epoch, 'Iteration:', batch_idx)
            print('Saving checkpoint file [' + checkpoint_filename + ']')


    def load_checkpoint(self, checkpoint_file=None):
        """Load a checkpoint."""

        checkpoint = torch.load(checkpoint_file)

        if "model" in checkpoint:
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint["model"].items() if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.epoch_count = checkpoint['epoch']
        self.step_count = checkpoint['total_step_count']
        self.checkpoint_batch_idx = checkpoint['batch_idx']

        print(f"epoch: {checkpoint['epoch']}, batch_idx: {checkpoint['batch_idx']}, batch_size: {checkpoint['batch_size']}, total_step_count: {checkpoint['total_step_count']}")

    def train(self, epoch):
        """Training process."""

        self.model.train()

        # Iterate over all batches in an epoch
        for step, batch in enumerate(self.train_data_loader, self.checkpoint_batch_idx):

            self.step_count += 1
            batch = {k: v.to(self.cfg.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            # >>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<
            # >>>>>>>>>>>>>> train on batch <<<<<<<<<<<<<<<<
            # >>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<
            # Get data from the batch
            images = batch['img']  # input image
            gt_keypoints_2d = batch['keypoints']  # 2D keypoints
            gt_pose = batch['pose']  # SMPL pose parameters
            gt_betas = batch['betas']  # SMPL beta parameters
            gt_joints = batch['pose_3d']  # 3D pose
            has_smpl = batch['has_smpl'].to(torch.bool)  # flag that indicates whether SMPL parameters are valid
            has_pose_3d = batch['has_pose_3d'].to(torch.bool)  # flag that indicates whether 3D pose is valid
            is_flipped = batch[
                'is_flipped']  # flag that indicates whether image was flipped during data augmentation
            rot_angle = batch['rot_angle']  # rotation angle used for data augmentation
            dataset_name = batch['dataset_name']  # name of the dataset the image comes from
            indices = batch['sample_index']  # index of example inside its dataset
            batch_size = images.shape[0]

            # Get GT vertices and model joints
            # Note that gt_model_joints is different from gt_joints as it comes from SMPL
            gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3])
            gt_model_joints = gt_out.joints
            gt_vertices = gt_out.vertices

            # Get current best fits from the dictionary
            opt_pose, opt_betas = self.fits_dict[(dataset_name, indices.cpu(), rot_angle.cpu(), is_flipped.cpu())]
            opt_pose = opt_pose.to(self.cfg.device)
            opt_betas = opt_betas.to(self.cfg.device)

            # Replace extreme betas with zero betas
            opt_betas[(opt_betas.abs() > 3).any(dim=-1)] = 0.
            # Replace the optimized parameters with the ground truth parameters, if available
            opt_pose[has_smpl, :] = gt_pose[has_smpl, :]
            opt_betas[has_smpl, :] = gt_betas[has_smpl, :]

            opt_output = self.smpl(betas=opt_betas, body_pose=opt_pose[:, 3:], global_orient=opt_pose[:, :3])
            opt_vertices = opt_output.vertices
            opt_joints = opt_output.joints

            batch['verts'] = opt_vertices

            # De-normalize 2D keypoints from [-1,1] to pixel space
            gt_keypoints_2d_orig = gt_keypoints_2d.clone()
            gt_keypoints_2d_orig[:, :, :-1] = 0.5 * self.cfg.IMG_RES * (gt_keypoints_2d_orig[:, :, :-1] + 1)

            # Estimate camera translation given the model joints and 2D keypoints
            # by minimizing a weighted least squares loss
            gt_cam_t = estimate_translation(gt_model_joints, gt_keypoints_2d_orig, focal_length=self.focal_length, img_size=self.cfg.IMG_RES)

            opt_cam_t = estimate_translation(opt_joints, gt_keypoints_2d_orig, focal_length=self.focal_length, img_size=self.cfg.IMG_RES)

            # get fitted smpl parameters as pseudo ground truth
            valid_fit = self.fits_dict.get_vaild_state(dataset_name, indices.cpu()).to(torch.bool).to(self.cfg.device)

            try:
                valid_fit = valid_fit | has_smpl
            except RuntimeError:
                valid_fit = (valid_fit.byte() | has_smpl.byte()).to(torch.bool)

            # Render Dense Correspondences
            if self.cfg.pymaf_model['AUX_SUPV_ON']:
                gt_cam_t_nr = opt_cam_t.detach().clone()
                gt_camera = torch.zeros(gt_cam_t_nr.shape).to(gt_cam_t_nr.device)
                gt_camera[:, 1:] = gt_cam_t_nr[:, :2]
                gt_camera[:, 0] = (2. * self.focal_length / self.cfg.IMG_RES) / gt_cam_t_nr[:, 2]
                iuv_image_gt = torch.zeros(
                    (batch_size, 3, self.cfg.pymaf_model['DP_HEATMAP_SIZE'], self.cfg.pymaf_model['DP_HEATMAP_SIZE'])).to(self.cfg.device)
                if torch.sum(valid_fit.float()) > 0:
                    iuv_image_gt[valid_fit] = self.iuv_maker.verts2iuvimg(opt_vertices[valid_fit], cam=gt_camera[valid_fit])  # [B, 3, 56, 56]
                batch['iuv_image_gt'] = iuv_image_gt

                uvia_list = iuv_img2map(iuv_image_gt)

            # Feed images in the network to predict camera and SMPL parameters
            preds_dict, _ = self.model(images)

            output = preds_dict
            loss_dict = {}

            if self.cfg.pymaf_model['AUX_SUPV_ON']:
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
                    loss_dict[f'loss_U{r_i}'] = loss_U
                    loss_dict[f'loss_V{r_i}'] = loss_V
                    loss_dict[f'loss_IndexUV{r_i}'] = loss_IndexUV
                    loss_dict[f'loss_segAnn{r_i}'] = loss_segAnn

            len_loop = len(preds_dict['smpl_out'])

            for l_i in range(len_loop):

                if l_i == 0:
                    # initial parameters (mean poses)
                    continue
                pred_rotmat = preds_dict['smpl_out'][l_i]['rotmat']
                pred_betas = preds_dict['smpl_out'][l_i]['theta'][:, 3:13]
                pred_camera = preds_dict['smpl_out'][l_i]['theta'][:, :3]

                pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                        global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
                pred_vertices = pred_output.vertices
                pred_joints = pred_output.joints

                # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
                # This camera translation can be used in a full perspective projection
                pred_cam_t = torch.stack([pred_camera[:, 1],
                                          pred_camera[:, 2],
                                          2 * self.focal_length / (self.cfg.IMG_RES * pred_camera[:, 0] + 1e-9)],
                                         dim=-1)

                camera_center = torch.zeros(batch_size, 2, device=self.cfg.device)
                pred_keypoints_2d = perspective_projection(pred_joints,
                                                           rotation=torch.eye(3, device=self.cfg.device).unsqueeze(
                                                               0).expand(batch_size, -1, -1),
                                                           translation=pred_cam_t,
                                                           focal_length=self.focal_length,
                                                           camera_center=camera_center)
                # Normalize keypoints to [-1,1]
                pred_keypoints_2d = pred_keypoints_2d / (self.cfg.IMG_RES / 2.)

                # Compute loss on SMPL parameters
                loss_regr_pose, loss_regr_betas = smpl_losses(pred_rotmat, pred_betas, opt_pose, opt_betas, valid_fit)
                loss_regr_pose *= self.cfg.LOSS['POSE_W']
                loss_regr_betas *= self.cfg.LOSS['SHAPE_W']
                loss_dict['loss_regr_pose_{}'.format(l_i)] = loss_regr_pose
                loss_dict['loss_regr_betas_{}'.format(l_i)] = loss_regr_betas

                # Compute 2D reprojection loss for the keypoints
                if self.cfg.LOSS['KP_2D_W'] > 0:
                    loss_keypoints = keypoint_loss(pred_keypoints_2d, gt_keypoints_2d,
                                                        self.cfg.openpose_train_weight,
                                                        self.cfg.gt_train_weight) * self.cfg.LOSS['KP_2D_W']
                    loss_dict['loss_keypoints_{}'.format(l_i)] = loss_keypoints

                # Compute 3D keypoint loss
                loss_keypoints_3d = keypoint_3d_loss(pred_joints, gt_joints, has_pose_3d) * self.cfg.LOSS['KP_3D_W']
                loss_dict['loss_keypoints_3d_{}'.format(l_i)] = loss_keypoints_3d

                # Per-vertex loss for the shape
                if self.cfg.LOSS['VERT_W'] > 0:
                    loss_shape = shape_loss(pred_vertices, opt_vertices, valid_fit) * self.cfg.LOSS['VERT_W']
                    loss_dict['loss_shape_{}'.format(l_i)] = loss_shape

                # Camera
                # force the network to predict positive depth values
                loss_cam = ((torch.exp(-pred_camera[:, 0] * 10)) ** 2).mean()
                loss_dict['loss_cam_{}'.format(l_i)] = loss_cam

            for key in loss_dict:
                if len(loss_dict[key].shape) > 0:
                    loss_dict[key] = loss_dict[key][0]

            # Compute total loss
            loss = torch.stack(list(loss_dict.values())).sum()

            # Do backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Pack output arguments for tensorboard logging
            output.update({'pred_vertices': pred_vertices.detach(),
                           'opt_vertices': opt_vertices,
                           'pred_cam_t': pred_cam_t.detach(),
                           'opt_cam_t': opt_cam_t})
            loss_dict['loss'] = loss.detach().item()

            for loss_name, val in loss_dict.items():
                self.summary.add_scalar('losses/{}'.format(loss_name), val, self.step_count)

            out = {'preds': output, 'losses': loss_dict}
            
            # Tensorboard logging every summary_steps steps
            if self.step_count % self.cfg.TRAIN_VIS_ITER_FERQ == 0:
                self.model.eval()
                self.visualize(self.step_count, batch, 'train', **out)
                self.model.train()

    def evaluate(self):
        if self.cfg.TRAIN_VAL_LOOP:
            step = self.cfg.pymaf_model['N_ITER'] + 1
        else:
            step = 1

        num_poses = len(self.evaluation_accumulators['pred_j3d']) * self.cfg.TRAIN_BATCHSIZE // step
        print(f'Evaluating on {num_poses} number of poses ...')

        for loop_id in range(step):
            pred_j3ds = self.evaluation_accumulators['pred_j3d'][loop_id::step]
            pred_j3ds = np.vstack(pred_j3ds)
            pred_j3ds = torch.from_numpy(pred_j3ds).float()

            target_j3ds = self.evaluation_accumulators['target_j3d'][loop_id::step]
            target_j3ds = np.vstack(target_j3ds)
            target_j3ds = torch.from_numpy(target_j3ds).float()

            # Absolute error (MPJPE)
            errors = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            S1_hat = compute_similarity_transform_batch(pred_j3ds.numpy(), target_j3ds.numpy())
            S1_hat = torch.from_numpy(S1_hat).float()
            errors_pa = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

            pred_verts = self.evaluation_accumulators['pred_verts'][loop_id::step]
            pred_verts = np.vstack(pred_verts)
            pred_verts = torch.from_numpy(pred_verts).float()

            target_verts = self.evaluation_accumulators['target_verts'][loop_id::step]
            target_verts = np.vstack(target_verts)
            target_verts = torch.from_numpy(target_verts).float()
            errors_pve = torch.sqrt(((pred_verts - target_verts) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

            m2mm = 1000
            pve = np.mean(errors_pve) * m2mm
            mpjpe = np.mean(errors) * m2mm
            pa_mpjpe = np.mean(errors_pa) * m2mm

            eval_dict = {
                'mpjpe': mpjpe,
                'pa-mpjpe': pa_mpjpe,
                'pve': pve,
            }

            loop_id -= step  # to ensure the index of latest prediction is always -1
            log_str = f'Epoch {self.epoch_count}, step {loop_id}  '
            log_str += ' '.join([f'{k.upper()}: {v:.4f},' for k, v in eval_dict.items()])
            print(log_str)

            for k, v in eval_dict.items():
                self.summary.add_scalar(f'eval_error/{k}_{loop_id}', v, global_step=self.epoch_count)

        # empty accumulators
        for k, v in self.evaluation_accumulators.items():
            self.evaluation_accumulators[k].clear()

        return pa_mpjpe


    def validate(self):
        with torch.no_grad():
            self.model.eval()
            start = time.time()
            print('Start Validation.')

            # initialize
            for k, v in self.evaluation_accumulators.items():
                self.evaluation_accumulators[k] = []

            # Regressor for H36m joints
            J_regressor = torch.from_numpy(np.load(self.cfg.JOINT_REGRESSOR_H36M)).float()

            joint_mapper_gt = self.cfg.J24_TO_J17 if self.cfg.eval_dataset == 'mpi-inf-3dhp' else self.cfg.J24_TO_J14

            for i, target in enumerate(self.valid_loader):
                # Get GT vertices and model joints
                gt_betas = target['betas'].to(self.cfg.device)
                gt_pose = target['pose'].to(self.cfg.device)
                gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3])
                gt_model_joints = gt_out.joints
                gt_vertices = gt_out.vertices
                target['verts'] = gt_vertices.cpu()

                inp = target['img'].to(self.cfg.device, non_blocking=True)
                J_regressor_batch = J_regressor[None, :].expand(inp.shape[0], -1, -1).contiguous().to(self.cfg.device, non_blocking=True)

                pred_dict, _ = self.model(inp, J_regressor=J_regressor_batch)

                if self.cfg.TRAIN_VAL_LOOP:
                    preds_list = pred_dict['smpl_out']
                else:
                    preds_list = pred_dict['smpl_out'][-1:]

                for preds in preds_list:
                    # convert to 14 keypoint format for evaluation
                    n_kp = preds['kp_3d'].shape[-2]
                    pred_j3d = preds['kp_3d'].view(-1, n_kp, 3).cpu().numpy()

                    target_j3d = target['pose_3d'].numpy()
                    target_j3d = target_j3d[:, joint_mapper_gt, :-1]

                    pred_verts = preds['verts'].cpu().numpy()
                    target_verts = target['verts'].numpy()

                    batch_len = target['betas'].shape[0]

                    self.evaluation_accumulators['pred_verts'].append(pred_verts)
                    self.evaluation_accumulators['target_verts'].append(target_verts)
                    self.evaluation_accumulators['pred_j3d'].append(pred_j3d)
                    self.evaluation_accumulators['target_j3d'].append(target_j3d)

                if (i + 1) % self.cfg.VAL_VIS_BATCH_FREQ == 0:
                    self.visualize(i, target, 'valid', pred_dict)

                del pred_dict, _

                batch_time = time.time() - start

    def visualize(self, it, target, stage, preds, losses=None):
        with torch.no_grad():
            theta = preds['smpl_out'][-1]['theta']
            pred_verts = preds['smpl_out'][-1]['verts'].cpu().numpy() if 'verts' in preds['smpl_out'][-1] else None
            cam_pred = theta[:, :3].detach()

            dp_out = preds['dp_out'][-1] if self.cfg.pymaf_model['AUX_SUPV_ON'] else None

            images = target['img']
            images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
            images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)
            imgs_np = images.cpu().numpy()

            vis_img_full = []
            vis_n = min(len(theta), 16)
            vis_img = []
            for b in range(vis_n):
                cam_t = cam_pred[b].cpu().numpy()
                smpl_verts = target['verts'][b].cpu().numpy()
                smpl_verts_pred = pred_verts[b] if pred_verts is not None else None

                render_imgs = []

                img_vis = np.transpose(imgs_np[b], (1, 2, 0)) * 255
                img_vis = img_vis.astype(np.uint8)

                render_imgs.append(img_vis)

                render_imgs.append(self.renderer(
                    smpl_verts,
                    self.smpl.faces,
                    image=img_vis,
                    cam=cam_t,
                    addlight=True
                ))

                if self.cfg.pymaf_model['AUX_SUPV_ON']:
                    if stage == 'train':
                        iuv_image_gt = target['iuv_image_gt'][b].detach().cpu().numpy()
                        iuv_image_gt = np.transpose(iuv_image_gt, (1, 2, 0)) * 255
                        iuv_image_gt_resized = resize(iuv_image_gt, (img_vis.shape[0], img_vis.shape[1]),
                                                      preserve_range=True, anti_aliasing=True)
                        render_imgs.append(iuv_image_gt_resized.astype(np.uint8))

                    pred_iuv_list = [dp_out['predict_u'][b:b + 1], dp_out['predict_v'][b:b + 1], dp_out['predict_uv_index'][b:b + 1], dp_out['predict_ann_index'][b:b + 1]]
                    iuv_image_pred = iuv_map2img(*pred_iuv_list)[0].detach().cpu().numpy()
                    iuv_image_pred = np.transpose(iuv_image_pred, (1, 2, 0)) * 255
                    iuv_image_pred_resized = resize(iuv_image_pred, (img_vis.shape[0], img_vis.shape[1]),
                                                    preserve_range=True, anti_aliasing=True)
                    render_imgs.append(iuv_image_pred_resized.astype(np.uint8))

                if smpl_verts_pred is not None:
                    render_imgs.append(self.renderer(
                        smpl_verts_pred,
                        self.smpl.faces,
                        image=img_vis,
                        cam=cam_t,
                        addlight=True
                    ))

                img = np.concatenate(render_imgs, axis=1)
                img = np.transpose(img, (2, 0, 1))
                vis_img.append(img)

            vis_img_full.append(np.concatenate(vis_img, axis=1))

            vis_img_full = np.concatenate(vis_img_full, axis=-1)
            if stage == 'train':
                self.summary.add_image('{}/mesh_pred'.format(stage), vis_img_full, it)
            else:
                self.summary.add_image('{}/mesh_pred_{}'.format(stage, it), vis_img_full, self.epoch_count)



    def run(self):
        for epoch in tqdm(range(self.epoch_count, self.cfg.epoch_num), total=self.cfg.epoch_num, initial=self.epoch_count):
            self.epoch_count = epoch
            self.train(epoch)

            self.validate()

            performance = self.evaluate()
            # log the learning rate
            for param_group in self.optimizer.param_groups:
                print(f'Learning rate {param_group["lr"]}')
                self.summary.add_scalar('lr/model_lr', param_group['lr'], global_step=self.epoch_count)

            is_best = performance < self.best_performance
            if is_best:
                print('Best performance achived, saved it!')
                self.best_performance = performance
            self.save_checkpoint(self.model, self.optimizer, epoch + 1, batch_idx=0, batch_size=self.cfg.TRAIN_BATCHSIZE,
                                       total_step_count=self.step_count, is_best=is_best)



