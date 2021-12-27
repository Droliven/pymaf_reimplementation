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

from torch.optim import Adam
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm
from pprint import pprint
import random
import numpy as np
import json


class RunnerPymaf():
    def __init__(self, exp_name="", device="cuda:0", num_works=0, is_debug=False, args=None):
        super(RunnerPymaf, self).__init__()

        self.is_debug = is_debug

        # 参数
        self.start_epoch = 1

        self.cfg = ConfigPymaf(exp_name=exp_name, device=device, num_works=num_works)

        print("\n================== Configs =================")
        pprint(vars(self.cfg), indent=4)
        print("==========================================\n")

        save_dict = {"args": args.__dict__, "cfgs": self.cfg.__dict__}
        save_json = json.dumps(save_dict)

        with open(os.path.join(self.cfg.output_dir, "config.json"), 'w', encoding='utf-8') as f:
            f.write(save_json)

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
        if args.single_dataset:
            self.train_ds = BaseDataset(eval_pve=self.cfg.eval_pve, noise_factor=self.cfg.noise_factor, rot_factor=self.cfg.rot_factor, scale_factor=self.cfg.scale_factor, dataset=self.cfg.single_dataname, ignore_3d=False, use_augmentation=True, is_train=True, DATASET_FOLDERS=self.cfg.DATASET_FOLDERS, DATASET_FILES=self.cfg.DATASET_FILES, JOINT_MAP=self.cfg.JOINT_MAP, JOINT_NAMES=self.cfg.JOINT_NAMES, J24_TO_J19=self.cfg.J24_TO_J19, JOINT_REGRESSOR_TRAIN_EXTRA=self.cfg.JOINT_REGRESSOR_TRAIN_EXTRA, SMPL_MODEL_DIR=self.cfg.SMPL_MODEL_DIR, IMG_NORM_MEAN=self.cfg.IMG_NORM_MEAN, IMG_NORM_STD=self.cfg.IMG_NORM_STD, TRAIN_BATCH_SIZE=self.cfg.TRAIN_BATCHSIZE, IMG_RES=self.cfg.IMG_RES, SMPL_JOINTS_FLIP_PERM=self.cfg.SMPL_JOINTS_FLIP_PERM)
        else:
            self.train_ds = MixedDataset(eval_pve=self.cfg.eval_pve, noise_factor=self.cfg.noise_factor, rot_factor=self.cfg.rot_factor, scale_factor=self.cfg.scale_factor, ignore_3d=False, use_augmentation=True, is_train=True, DATASET_FOLDERS=self.cfg.DATASET_FOLDERS, DATASET_FILES=self.cfg.DATASET_FILES, JOINT_MAP=self.cfg.JOINT_MAP, JOINT_NAMES=self.cfg.JOINT_NAMES, J24_TO_J19=self.cfg.J24_TO_J19, JOINT_REGRESSOR_TRAIN_EXTRA=self.cfg.JOINT_REGRESSOR_TRAIN_EXTRA, SMPL_MODEL_DIR=self.cfg.SMPL_MODEL_DIR, IMG_NORM_MEAN=self.cfg.IMG_NORM_MEAN, IMG_NORM_STD=self.cfg.IMG_NORM_STD, TRAIN_BATCH_SIZE=self.cfg.TRAIN_BATCHSIZE, IMG_RES=self.cfg.IMG_RES, SMPL_JOINTS_FLIP_PERM=self.cfg.SMPL_JOINTS_FLIP_PERM)

        self.valid_ds = BaseDataset(eval_pve=self.cfg.eval_pve, noise_factor=self.cfg.noise_factor, rot_factor=self.cfg.rot_factor, scale_factor=self.cfg.scale_factor, dataset=self.cfg.eval_dataset, ignore_3d=False, use_augmentation=True, is_train=False, DATASET_FOLDERS=self.cfg.DATASET_FOLDERS, DATASET_FILES=self.cfg.DATASET_FILES, JOINT_MAP=self.cfg.JOINT_MAP, JOINT_NAMES=self.cfg.JOINT_NAMES, J24_TO_J19=self.cfg.J24_TO_J19, JOINT_REGRESSOR_TRAIN_EXTRA=self.cfg.JOINT_REGRESSOR_TRAIN_EXTRA, SMPL_MODEL_DIR=self.cfg.SMPL_MODEL_DIR, IMG_NORM_MEAN=self.cfg.IMG_NORM_MEAN, IMG_NORM_STD=self.cfg.IMG_NORM_STD, TRAIN_BATCH_SIZE=self.cfg.TRAIN_BATCHSIZE, IMG_RES=self.cfg.IMG_RES, SMPL_JOINTS_FLIP_PERM=self.cfg.SMPL_JOINTS_FLIP_PERM)

        self.train_data_loader = DataLoader(
            self.train_ds,
            batch_size=self.cfg.TRAIN_BATCHSIZE,
            num_workers=self.cfg.workers,
            pin_memory=True,
            shuffle=True,
        )

        self.valid_loader = DataLoader(
            dataset=self.valid_ds,
            batch_size=self.cfg.TEST_BATCHSIZE,
            shuffle=False,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
        )

        # Load dictionary of fits
        self.fits_dict = FitsDict(self.options, self.train_ds)
        self.evaluation_accumulators = dict.fromkeys(
            ['pred_j3d', 'target_j3d', 'target_theta', 'pred_verts', 'target_verts'])

        # Create renderer
        try:
            self.renderer = OpenDRenderer()
        except:
            print('No renderer for visualization.')
            self.renderer = None

        if self.cfg.pymaf_model['AUX_SUPV_ON']:
            self.iuv_maker = IUV_Renderer(output_size=cfg.MODEL.PyMAF.DP_HEATMAP_SIZE)

        self.decay_steps_ind = 1
        self.decay_epochs_ind = 1

        # log
        self.summary = SummaryWriter(self.cfg.output_dir)


    def finalize(self):
        self.fits_dict.save()

    def save(self, checkpoint_path, epoch, curr_err):
        state = {
            "epoch": epoch,
            "lr": self.scheduler.get_last_lr()[0],
            "curr_err": curr_err,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, checkpoint_path)

    def restore(self, checkpoint_path):
        state = torch.load(checkpoint_path, map_location=self.cfg.device)
        self.model.load_state_dict(state["model"])
        # self.optimizer.load_state_dict(state["optimizer"])
        # self.lr = state["lr"]
        # self.start_epoch = state["epoch"] + 1
        curr_err = state["curr_err"]
        print(
            "load from epoch {}, lr {}, curr_avg {}.".format(state["epoch"], self.scheduler.get_last_lr()[0], curr_err))

    def train(self, epoch, draw=False):
        self.model.train()

        average_allloss = 0
        average_kls_p1 = 0
        average_kls_p2 = 0

        average_adeerrors = 0
        average_mmadeerrors = 0
        average_hinges = 0

        average_his = 0
        average_limblen = 0
        average_angle = 0

        dg = self.train_data.batch_generator()
        generator_len = self.cfg.sub_len_train // self.train_data.batch_size if not self.is_debug else 200 // self.train_data.batch_size
        draw_i = random.randint(0, generator_len - 1)
        for i, (datas, similars) in enumerate(dg):
            # [b, 48, 125], [b, 10, 48, 125]
            b, vc, t = datas.shape
            # skip the last batch if only have one sample for batch_norm layers
            if b == 1:
                continue

            self.global_step = (epoch - 1) * generator_len + i + 1

            datas = torch.from_numpy(datas).float().cuda(device=self.cfg.device)
            similars = torch.from_numpy(similars).float().cuda(device=self.cfg.device)
            eps = torch.randn((b, self.cfg.z_dim), device=self.cfg.device)
            with torch.no_grad():
                padded_inputs = datas[:, :, list(range(self.cfg.t_his)) + [self.cfg.t_his - 1] * self.cfg.t_pred]
                padded_inputs_dct = dct_transform_torch(padded_inputs, self.dct_m, dct_n=self.cfg.dct_n)  # b, 48, 10
                padded_inputs_dct = padded_inputs_dct.view(b, -1, 3 * self.cfg.dct_n)  # # b, 16, 3*10

            # train generator
            repeated_eps = torch.repeat_interleave(eps, repeats=self.cfg.part_1_head_train, dim=0)
            all_z_p1, all_mean_p1, all_logvar_p1 = self.model(condition=padded_inputs_dct, repeated_eps=repeated_eps, temperature=self.cfg.temperature_p1, multi_modal_head=self.cfg.part_1_head_train, mode="p1")  # b*(10), 128
            repeated_eps = torch.repeat_interleave(eps, repeats=self.cfg.train_nk - self.cfg.part_1_head_train, dim=0)
            all_z_p2, all_mean_p2, all_logvar_p2 = self.model(condition=padded_inputs_dct, repeated_eps=repeated_eps, temperature=self.cfg.temperature_p2, multi_modal_head=self.cfg.train_nk - self.cfg.part_1_head_train, mode="p2")  # b*(90), 128

            all_z_p1 = all_z_p1.view(b, self.cfg.part_1_head_train, self.cfg.z_dim)
            all_z_p2 = all_z_p2.view(b, self.cfg.train_nk - self.cfg.part_1_head_train, self.cfg.z_dim)
            all_z = torch.cat((all_z_p1, all_z_p2), dim=1).view(-1, self.cfg.z_dim)

            all_outs_dct = self.model_t1.inference(condition=torch.repeat_interleave(padded_inputs_dct, repeats=self.cfg.train_nk, dim=0), z=all_z) # b*h, 16, 30
            all_outs_dct = all_outs_dct.reshape(b*self.cfg.train_nk, -1, self.cfg.dct_n)  # b*h, 48, 10
            outputs = reverse_dct_torch(all_outs_dct, self.i_dct_m, self.cfg.t_total)  # b*h, 48, 125
            outputs = outputs.view(b, self.cfg.train_nk, -1, self.cfg.t_total) # b, 50, 48, 125

            # loss
            kls_p1 = loss_kl_normal(all_mean_p1, all_logvar_p1)
            kls_p2 = loss_kl_normal(all_mean_p2, all_logvar_p2)
            adeerrors = loss_recons_adelike(gt=datas[:, :, self.cfg.t_his:], pred=outputs[:, :self.cfg.part_1_head_train, :, self.cfg.t_his:])
            mmadeerrors = loss_recons_mmadelike(similars=similars[:, :, :, self.cfg.t_his:], pred=outputs[:, :self.cfg.part_1_head_train, :, self.cfg.t_his:])

            all_hinges = []
            for oidx in range(self.cfg.train_nk // self.cfg.seperate_head):
                all_hinges.append(loss_diversity_hinge(outputs[:, oidx*self.cfg.seperate_head:(oidx+1)*self.cfg.seperate_head, :, self.cfg.t_his:], minthreshold=self.cfg.minthreshold))  # [10, 48, 100], [1, 48, 100]
                # all_hinges.append(loss_diversity_hinge_v2_likedlow(outputs[:, oidx*self.cfg.seperate_head:(oidx+1)*self.cfg.seperate_head, :, self.cfg.t_his:], minthreshold=225))  # [10, 48, 100], [1, 48, 100]
                for ojdx in range(oidx + 1, self.cfg.train_nk // self.cfg.seperate_head):
                    all_hinges.append(loss_diversity_hinge_between_two_part(outputs[:, oidx*self.cfg.seperate_head:(oidx+1)*self.cfg.seperate_head, :, self.cfg.t_his:], outputs[:, ojdx*self.cfg.seperate_head:(ojdx+1)*self.cfg.seperate_head, :, self.cfg.t_his:], minthreshold=self.cfg.minthreshold))  # [10, 48, 100], [1, 48, 100]
                    # all_hinges.append(loss_diversity_hinge_between_two_part_v2_likedlow(outputs[:, oidx*self.cfg.seperate_head:(oidx+1)*self.cfg.seperate_head, :, self.cfg.t_his:], outputs[:, ojdx*self.cfg.seperate_head:(ojdx+1)*self.cfg.seperate_head, :, self.cfg.t_his:], minthreshold=225))  # [10, 48, 100], [1, 48, 100]
            all_hinges = torch.cat(all_hinges, dim=-1).mean(dim=-1).mean()

            # all_hinges = []
            # for oidx in range((self.cfg.train_nk - self.cfg.part_1_head_train) // self.cfg.seperate_head):
            #     all_hinges.append(loss_diversity_hinge(
            #         outputs[:, oidx * self.cfg.seperate_head + self.cfg.part_1_head_train:(oidx + 1) * self.cfg.seperate_head + self.cfg.part_1_head_train, :, self.cfg.t_his:],
            #         minthreshold=self.cfg.minthreshold))  # [10, 48, 100], [1, 48, 100]
            #     # all_hinges.append(loss_diversity_hinge_v2_likedlow(outputs[:, oidx*self.cfg.seperate_head:(oidx+1)*self.cfg.seperate_head, :, self.cfg.t_his:], minthreshold=225))  # [10, 48, 100], [1, 48, 100]
            #     for ojdx in range(oidx + 1, (self.cfg.train_nk - self.cfg.part_1_head_train) // self.cfg.seperate_head):
            #         all_hinges.append(loss_diversity_hinge_between_two_part(
            #             outputs[:, oidx * self.cfg.seperate_head+ self.cfg.part_1_head_train:(oidx + 1) * self.cfg.seperate_head+ self.cfg.part_1_head_train, :,
            #             self.cfg.t_his:],
            #             outputs[:, ojdx * self.cfg.seperate_head+ self.cfg.part_1_head_train:(ojdx + 1) * self.cfg.seperate_head+ self.cfg.part_1_head_train, :,
            #             self.cfg.t_his:], minthreshold=self.cfg.minthreshold))  # [10, 48, 100], [1, 48, 100]
            #         # all_hinges.append(loss_diversity_hinge_between_two_part_v2_likedlow(outputs[:, oidx*self.cfg.seperate_head:(oidx+1)*self.cfg.seperate_head, :, self.cfg.t_his:], outputs[:, ojdx*self.cfg.seperate_head:(ojdx+1)*self.cfg.seperate_head, :, self.cfg.t_his:], minthreshold=225))  # [10, 48, 100], [1, 48, 100]
            # all_hinges = torch.cat(all_hinges, dim=-1).mean(dim=-1).mean()

            recovrhis_err = 0 # loss_recover_history_t2(outputs[:, :, :, :self.cfg.t_his], datas[:, :, :self.cfg.t_his])  # 这里用 25 帧
            limblen_err = 0 # loss_limb_length_t2(outputs, datas, parent_17=self.cfg.parents_17)  # 这里用 125帧
            angle_err = 0 # loss_valid_angle_t2(outputs[:, :, :, self.cfg.t_his:], self.valid_angle)

            all_loss = kls_p1 * self.cfg.t2_kl_p1_weight \
                       + adeerrors * self.cfg.t2_ade_weight \
                       + mmadeerrors * self.cfg.t2_recons_mm_weight \
                       + kls_p2 * self.cfg.t2_kl_p2_weight \
                       + all_hinges * self.cfg.t2_diversity_weight
                       # + limblen_err * self.cfg.t2_limblen_weight \
                       # + recovrhis_err * self.cfg.t2_recoverhis_weight

            if angle_err > 0:
                all_loss += angle_err * self.cfg.t2_angle_weight

            self.optimizer.zero_grad()
            all_loss.backward()
            # grad_norm = torch.nn.utils.clip_grad_norm_(list(self.model.parameters()), max_norm=100)
            self.optimizer.step()

            average_allloss += all_loss
            average_kls_p1 += kls_p1
            average_kls_p2 += kls_p2
            average_adeerrors += adeerrors
            average_mmadeerrors += mmadeerrors
            average_hinges += all_hinges

            average_his += recovrhis_err
            average_limblen += limblen_err
            average_angle += angle_err

            # 画图
            if draw:
                if i == draw_i:
                    bidx = 0
                    origin = datas[bidx:bidx + 1].detach().cpu().numpy()  # 1, 48, 125
                    origin = origin.reshape(1, -1, 3, self.cfg.t_total)  # 1, 16, 3, 125
                    origin = np.concatenate((np.expand_dims(np.mean(origin[:, [0, 3], :, :], axis=1), axis=1), origin),
                                            axis=1)  # # 1, 17, 3, 125
                    origin *= 1000

                    output = outputs[bidx, :self.cfg.seperate_head].reshape(self.cfg.seperate_head, -1, 3, self.cfg.t_total).detach().cpu().numpy()  # 50, 16, 3, 125
                    output = np.concatenate((np.expand_dims(np.mean(output[:, [0, 3], :, :], axis=1), axis=1), output),
                                            axis=1)  # # 50, 17, 3, 100
                    output *= 1000

                    all_to_draw = np.concatenate((origin, output), axis=0)
                    draw_acc = [acc for acc in range(0, all_to_draw.shape[-1], 5)]
                    all_to_draw = all_to_draw[:, :, :, draw_acc][:, :, [0, 2], :]

                    draw_multi_seqs_2d(all_to_draw, gt_cnt=1, t_his=5,
                                       I=self.cfg.I17_plot, J=self.cfg.J17_plot,
                                       LR=self.cfg.LR17_plot,
                                       full_path=os.path.join(self.cfg.ckpt_dir, "images",
                                                              f"train_epo{epoch}idx{draw_i}.png"))

        average_allloss /= (i + 1)
        average_kls_p1 /= (i + 1)
        average_kls_p2 /= (i + 1)
        average_adeerrors /= (i + 1)
        average_mmadeerrors /= (i + 1)
        average_hinges /= (i + 1)
        average_his /= (i + 1)
        average_limblen /= (i + 1)
        average_angle /= (i + 1)

        self.summary.add_scalar("loss/average_all", average_allloss, epoch)
        self.summary.add_scalar("loss/average_kls_p1", average_kls_p1, epoch)
        self.summary.add_scalar("loss/average_kls_p2", average_kls_p2, epoch)
        self.summary.add_scalar("loss/average_ades", average_adeerrors, epoch)
        self.summary.add_scalar("loss/average_mmades", average_mmadeerrors, epoch)
        self.summary.add_scalar("loss/average_hinges", average_hinges, epoch)
        self.summary.add_scalar(f"loss/averagerhis", average_his, epoch)
        self.summary.add_scalar(f"loss/averagelimblen", average_limblen, epoch)
        self.summary.add_scalar(f"loss/averageangle", average_angle, epoch)
        return average_allloss, average_adeerrors, average_mmadeerrors, average_hinges, average_kls_p1, average_kls_p2, average_his, average_limblen, average_angle

    def eval(self, epoch=-1, draw=False):
        self.model.eval()

        diversity = 0
        ade = 0
        fde = 0
        mmade = 0
        mmfde = 0
        # 画图 ------------------------------------------------------------------------------------------------------
        if not os.path.exists(os.path.join(self.cfg.ckpt_dir, "images", "sample")):
            os.makedirs(os.path.join(self.cfg.ckpt_dir, "images", "sample"))

        dg = self.test_data.onebyone_generator()
        generator_len = len(self.test_data.similat_gt_like_dlow) if not self.is_debug else 90

        draw_i = random.randint(0, generator_len - 1)

        for i, datas in enumerate(dg):
            # b, 48, 125
            b, vc, t = datas.shape
            similars = self.test_data.similat_gt_like_dlow[i]  # 0/n, 48, 100
            if similars.shape[0] == 0:  # todo 这会淡化误差
                continue

            datas = torch.from_numpy(datas).float().cuda(device=self.cfg.device)
            similars = torch.from_numpy(similars).float().cuda(device=self.cfg.device)
            # eps = torch.randn((b, self.cfg.z_dim), device=self.cfg.device)

            with torch.no_grad():
                padded_inputs = datas[:, :, list(range(self.cfg.t_his)) + [self.cfg.t_his - 1] * self.cfg.t_pred]
                padded_inputs_dct = dct_transform_torch(padded_inputs, self.dct_m, dct_n=self.cfg.dct_n)  # b, 48, 10
                padded_inputs_dct = padded_inputs_dct.view(b, -1, 3 * self.cfg.dct_n)  # # b, 16, 3*10

                # train generator
                repeated_eps_1 = torch.randn((b * self.cfg.part_1_head_test, self.cfg.z_dim), device=self.cfg.device)
                all_z_p1, all_mean_p1, all_logvar_p1 = self.model(condition=padded_inputs_dct, repeated_eps=repeated_eps_1,
                                                                  temperature=self.cfg.temperature_p1,
                                                                  multi_modal_head=self.cfg.part_1_head_test,
                                                                  mode="p1")  # b*(10), 128
                repeated_eps_2 = torch.randn((b * (self.cfg.test_nk - self.cfg.part_1_head_test), self.cfg.z_dim), device=self.cfg.device)
                all_z_p2, all_mean_p2, all_logvar_p2 = self.model(condition=padded_inputs_dct, repeated_eps=repeated_eps_2,
                                                                  temperature=self.cfg.temperature_p2,
                                                                  multi_modal_head=self.cfg.test_nk - self.cfg.part_1_head_test,
                                                                  mode="p2")  # b*(90), 128
                all_z_p1 = all_z_p1.view(b, self.cfg.part_1_head_test, self.cfg.z_dim)
                all_z_p2 = all_z_p2.view(b, self.cfg.test_nk - self.cfg.part_1_head_test, self.cfg.z_dim)
                all_z = torch.cat((all_z_p1, all_z_p2), dim=1).view(-1, self.cfg.z_dim)

                all_outs_dct = self.model_t1.inference(condition=torch.repeat_interleave(padded_inputs_dct, repeats=self.cfg.test_nk, dim=0), z=all_z)  # b*h, 16, 30
                all_outs_dct = all_outs_dct.reshape(b*self.cfg.test_nk, -1, self.cfg.dct_n)  # b*h, 48, 10
                outputs = reverse_dct_torch(all_outs_dct, self.i_dct_m, self.cfg.t_total)  # b*h, 48, 125
                outputs = outputs.view(self.cfg.test_nk, -1, self.cfg.t_total)[:, :, self.cfg.t_his:]  # 50, 48, 100

                cdiv = compute_diversity(outputs).mean()  # [10, 48, 100], [1, 48, 100]
                cade = compute_ade(outputs, datas[:, :, self.cfg.t_his:])
                cfde = compute_fde(outputs, datas[:, :, self.cfg.t_his:])
                cmmade = compute_mmade(outputs, datas[:, :, self.cfg.t_his:], similars)
                cmmfde = compute_mmfde(outputs, datas[:, :, self.cfg.t_his:], similars)

            diversity += cdiv
            ade += cade
            fde += cfde
            mmade += cmmade
            mmfde += cmmfde
            if epoch == -1:
                print("Test --> it {}: div {:.4f} -- ade {:.4f} --  mmade {:.4f} --  fde {:.4f}  --  mmfde {:.4f} ".format(i, cdiv, cade, cmmade, cfde, cmmfde))

            if draw:
                if i == draw_i:
                    bidx = 0

                    origin = datas[bidx:bidx + 1].reshape(1, -1, 3,
                                                          self.cfg.t_total).cpu().data.numpy()  # 1, 16, 3, 125
                    origin = np.concatenate(
                        (np.expand_dims(np.mean(origin[:, [0, 3], :, :], axis=1), axis=1), origin),
                        axis=1)  # # 1, 17, 3, 125
                    origin *= 1000

                    all_outputs = outputs.cpu().data.numpy().reshape(self.cfg.test_nk, -1, 3, self.cfg.t_pred)  # 10, 16, 3, 100
                    all_outputs = np.concatenate(
                        (np.expand_dims(np.mean(all_outputs[:, [0, 3], :, :], axis=1), axis=1), all_outputs),
                        axis=1)  # 10, 17, 3, 100
                    all_outputs *= 1000
                    all_outputs = np.concatenate((np.repeat(origin[:, :, :, :self.cfg.t_his],
                                                            repeats=self.cfg.test_nk, axis=0), all_outputs),
                                                 axis=-1)  # 10, 17, 3, 125

                    all_to_draw = np.concatenate((origin, all_outputs), axis=0)  # 1 + 10, 17, 3, 125
                    draw_acc = [acc for acc in range(0, all_to_draw.shape[-1], 5)]
                    all_to_draw = all_to_draw[:, :, :, draw_acc][:, :, [0, 2], :]

                    draw_multi_seqs_2d(all_to_draw, gt_cnt=1, t_his=5, I=self.cfg.I17_plot,
                                       J=self.cfg.J17_plot,
                                       LR=self.cfg.LR17_plot,
                                       full_path=os.path.join(self.cfg.ckpt_dir, "images", "sample",
                                                              f"test_epo{epoch}idx{draw_i}.png"))

        diversity /= (i+1)
        ade /= (i+1)
        fde /= (i+1)
        mmade /= (i+1)
        mmfde /= (i+1)
        self.summary.add_scalar(f"Test/div", diversity, epoch)
        self.summary.add_scalar(f"Test/ade", ade, epoch)
        self.summary.add_scalar(f"Test/fde", fde, epoch)
        self.summary.add_scalar(f"Test/mmade", mmade, epoch)
        self.summary.add_scalar(f"Test/mmfde", mmfde, epoch)

        return diversity, ade, mmade, fde, mmfde

    def choose_eval(self, epoch=-1):
        self.model.eval()


        dg = self.test_data.onebyone_generator()
        choosed_idxs = [3589, 2538, 4433, 3028, 1192]
        for i, datas in enumerate(dg):
            if i in choosed_idxs:
                np.save(f"t1cvaegcndct_testidx{i}_gt.npy", datas)

                # b, 48, 125
                b, vc, t = datas.shape
                datas = torch.from_numpy(datas).float().cuda(device=self.cfg.device)
                eps = torch.randn((b, self.cfg.z_dim), device=self.cfg.device)
                with torch.no_grad():
                    padded_inputs = datas[:, :, list(range(self.cfg.t_his)) + [self.cfg.t_his - 1] * self.cfg.t_pred]
                    padded_inputs_dct = dct_transform_torch(padded_inputs, self.dct_m,
                                                            dct_n=self.cfg.dct_n)  # b, 48, 10
                    padded_inputs_dct = padded_inputs_dct.view(b, -1, 3 * self.cfg.dct_n)  # # b, 16, 3*10

                    # train generator
                    all_z, all_mean, all_logvar = self.model(condition=padded_inputs_dct, eps=eps,
                                                             temperature=self.cfg.temperature_p1,
                                                             multi_modal_head=self.cfg.test_nk)  # b*(10), 128
                    all_outs_dct = self.model_t1.inference(
                        condition=torch.repeat_interleave(padded_inputs_dct, repeats=self.cfg.test_nk, dim=0),
                        z=all_z)  # b*h, 16, 30
                    all_outs_dct = all_outs_dct.reshape(b * self.cfg.test_nk, -1, self.cfg.dct_n)  # b*h, 48, 10
                    outputs = reverse_dct_torch(all_outs_dct, self.i_dct_m, self.cfg.t_total)  # b*h, 48, 125
                    outputs = outputs.view(self.cfg.test_nk, -1, self.cfg.t_total)[:, :, self.cfg.t_his:]  # 50, 48, 100

                    outputs = outputs.cpu().data.numpy()
                    np.save(f"t2baseresamplegcndct_testidx{i}_sampletime{outputs.shape[0]}.npy", outputs)
                    print(f"{i}_{outputs.shape}")

    def run(self):
        for epoch in range(self.start_epoch, self.cfg.epoch_t2 + 1):
            self.summary.add_scalar("LR", self.scheduler.get_last_lr()[0], epoch)

            average_allloss, average_adeerrors, average_mmadeerrors, average_hinges, average_kls_p1, average_kls_p2, average_his, average_limblen, average_angle = self.train(epoch, draw=True)
            self.scheduler.step()

            print("Train >>> Epoch {}: all {:.4f} | ades {:.4f} |  mmades {:.4f} |  hinges {:.4f} |  klsp1 {:.4f} | klsp2 {:.4f} | losshis {:.6f} | losslimb {:.6f} |  lossangle {:.6f}".format(
                epoch, average_allloss, average_adeerrors, average_mmadeerrors, average_hinges, average_kls_p1, average_kls_p2, average_his, average_limblen, average_angle))

            if self.is_debug:
                test_interval = 1
            else:
                test_interval = 20

            if epoch % test_interval == 0:
                div, ade, mmade, fde, mmfde = self.eval(epoch=epoch, draw=True)
                print("Test --> epo {}: div {:.4f} -- ade {:.4f} --  mmade {:.4f} --  fde {:.4f}  --  mmfde {:.4f} ".format(epoch,
                                                                                                                     div,
                                                                                                                     ade,
                                                                                                                      mmade,
                                                                                                                      fde,
                                                                                                                     mmfde))

                self.save(os.path.join(self.cfg.ckpt_dir, "models", '{}_err{:.4f}.pth'.format(epoch, ade)), epoch, ade)

