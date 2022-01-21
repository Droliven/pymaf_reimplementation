#!/usr/bin/env python
# encoding: utf-8
'''
@project : pymaf_reimp
@file    : cfg_pymafreimp.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2022-01-21 09:43
'''
import os
import os.path as osp

import getpass

def get_cfg_pymafreimp(exp_name='', is_debug=True):

    cfg_json_dict = {

        'constants': {
            'focal_length': 5000.,
            'img_size': [224, 224],
            # Mean and standard deviation for normalizing input image
            'img_norm_mean': [0.485, 0.456, 0.406],
            'img_norm_std': [0.229, 0.224, 0.225],

            # We create a superset of joints containing the OpenPose joints together with the ones that each dataset provides.
            # We keep a superset of 24 joints such that we include all joints from every dataset.
            # If a dataset doesn't provide annotations for a specific joint, we simply ignore it.
            # The joints used here are the following:
            'joint_names': [
                # 25 OpenPose joints (in the order provided by OpenPose)
                'OP Nose',
                'OP Neck',
                'OP RShoulder',
                'OP RElbow',
                'OP RWrist',
                'OP LShoulder',
                'OP LElbow',
                'OP LWrist',
                'OP MidHip',
                'OP RHip',
                'OP RKnee',
                'OP RAnkle',
                'OP LHip',
                'OP LKnee',
                'OP LAnkle',
                'OP REye',
                'OP LEye',
                'OP REar',
                'OP LEar',
                'OP LBigToe',
                'OP LSmallToe',
                'OP LHeel',
                'OP RBigToe',
                'OP RSmallToe',
                'OP RHeel',
                # 24 Ground Truth joints (superset of joints from different datasets)
                'Right Ankle',
                'Right Knee',
                'Right Hip',  # 2
                'Left Hip',
                'Left Knee',  # 4
                'Left Ankle',
                'Right Wrist',  # 6
                'Right Elbow',
                'Right Shoulder',  # 8
                'Left Shoulder',
                'Left Elbow',  # 10
                'Left Wrist',
                'Neck (LSP)',  # 12
                'Top of Head (LSP)',
                'Pelvis (MPII)',  # 14
                'Thorax (MPII)',
                'Spine (H36M)',  # 16
                'Jaw (H36M)',
                'Head (H36M)',  # 18
                'Nose',
                'Left Eye',
                'Right Eye',
                'Left Ear',
                'Right Ear'
            ],
            # Dict containing the joints in numerical order
            'joint_ids': {},
            # Map joints to SMPL joints
            "joint_map": {
                'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
                'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
                'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
                'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
                'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
                'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
                'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
                'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
                'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
                'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
                'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
                'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
                'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
                'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
                'Spine (H36M)': 51, 'Jaw (H36M)': 52,
                'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
                'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
            },

            # Joint selectors, openpose 有 25 个点，SMPL 有 24 个点，H36M 有 17个点，LSP 有 14 个点，
            # Indices to get the 14 LSP joints from the 17 H36M joints
            "h36m_to_j17": [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9],
            "h36m_to_j14": [],
            # Indices to get the 14 LSP joints from the ground truth joints
            "j24_to_j17": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17],
            "j24_to_j14": [],
            "j24_to_j19": [],
            "j24_to_coco": [19, 20, 21, 22, 23, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0],

            # Permutation of SMPL pose parameters when flipping the shape
            "smpl_joints_flip_permutation": [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21,
                                             20, 23, 22],
            "smpl_pose_flip_permutation": [],

            # Permutation indices for the 24 ground truth joints
            "j24_flip_permutation": [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18, 19, 21, 20, 23,
                                     22],
            # Permutation indices for the full set of 49 joints
            "j49_flip_permutation": [],
            "smpl_j49_flip_permutation": [],

        },

        'dataset': {
            'train_list': [{
                'name': 'train_h36m',
                # 'annotation': osp.join("dataset_extras", 'h36m_train.npz'),
                'annotation': osp.join("dataset_extras", 'h36m_mosh_train.npz'),
                'img': osp.join('h36m/imgs')
            }, {
                'name': "train_lsporig",
                'annotation': osp.join("dataset_extras", 'lsp_dataset_original_train.npz'),
                'img': osp.join('lsp/lsp_dataset_original'),
            }, {
                'name': "train_mpii",
                'annotation': osp.join("dataset_extras", 'mpii_train.npz'),
                'img': osp.join('mpii'),
            }, {
                'name': "train_coco2014",
                'annotation': osp.join("dataset_extras", 'coco_2014_train.npz'),
                'img': osp.join('coco2014'),
            }, {
                'name': "train_lspet",
                'annotation': osp.join("dataset_extras", 'hr_lspet_train.npz'),
                'img': osp.join('lsp/hr_lspet'),
            }, {
                'name': "train_mpiinf3dhp",
                'annotation': osp.join("dataset_extras", 'mpi_inf_3dhp_train.npz'),
                'img': osp.join('mpi_inf_3dhp/mpi_inf_3dhp_train_set'),
            }],
            "test_list": [{
                'name': "test_h36mp1",
                'annotation': osp.join("dataset_extras", 'h36m_valid_protocol1.npz'),
                'img': osp.join('h36m/imgs'),
            }, {
                'name': "test_h36mp2",
                'annotation': osp.join("dataset_extras", 'h36m_valid_protocol2.npz'),
                'img': osp.join('h36m/imgs'),
            }, {
                'name': "test_h36mp2mosh",
                'annotation': osp.join("dataset_extras", 'h36m_mosh_valid_p2.npz'),
                'img': osp.join('h36m/imgs'),
            }, {
                'name': "test_lsp",
                'annotation': osp.join("dataset_extras", 'lsp_dataset_test.npz'),
                'img': osp.join('lsp/lsp_dataset_small')
            }, {
                'name': "test_mpiinf3dhp",
                'annotation': osp.join("dataset_extras", 'mpi_inf_3dhp_valid.npz'),
                'img': osp.join('mpi_inf_3dhp/mpi_inf_3dhp_test_set'),
            }, {
                'name': "test_3dpw",
                'annotation': osp.join("dataset_extras", '3dpw_test.npz'),
                'img': osp.join('3dpw'),
            }, {
                'name': "test_coco2014",
                'annotation': osp.join("dataset_extras", 'coco_2014_val.npz'),
                'img': osp.join('coco2014'),
            }],
        },

        'run': {
            'exp_name': exp_name,
            'is_debug': is_debug,
            'platform': getpass.getuser(),
            'device': 'cuda:0',
            'single_dataname': 'train_h36m',
            'eval_dataset': 'test_h36mp2mosh',
            'output_dir': osp.join(osp.abspath("./results"), exp_name),

            'num_works': 0,
        },
        'train': {
            'batch_size': 16,
            'gt_train_weight': 1.0,
            'openpose_train_weight': 0.0,

            'keypoint_loss_weight': 300,
            'pose_loss_weight': 60,
            'beta_loss_weight': 0.06,
            'shape_loss_weight': 0,
            'point_regression_weights': 0.5,

            'lr': 5e-5,
            'num_epochs': 60,

            'save_steps': 1000,
            'log_steps': 1000,
            'test_steps': 1000,

            'bn_momentum': 0.1,
            'noise_factor': 0.4,
            'rot_factor': 30,
            'scale_factor': 0.25,

            'train_with_loop': True,
            'backbone': {
                'name': 'res50',
                'mlp_dim': [256, 128, 64, 5],
                'n_iter': 3,
                'aux_supv_on': True,
                'dp_heatmap_size': 56,
            },
            'res_model': {
                'deconv_with_bias': False,
                'num_deconv_layers': 3,
                'num_deconv_filters': [256, 256, 256],
                'num_deconv_kernels': [4, 4, 4],
            }
        },
    }

    # 后处理
    cfg_json_dict["constants"]["joint_ids"] = {cfg_json_dict["constants"]["joint_names"][i]: i for i in
                                               range(len(cfg_json_dict["constants"]["joint_names"]))}
    cfg_json_dict["constants"]["h36m_to_j14"] = cfg_json_dict["constants"]["h36m_to_j17"][:14]
    cfg_json_dict["constants"]["j24_to_j14"] = cfg_json_dict["constants"]["j24_to_j17"][:14]
    cfg_json_dict["constants"]["j24_to_j19"] = cfg_json_dict["constants"]["j24_to_j14"] + [19, 20, 21, 22, 23]
    for i in cfg_json_dict["constants"]["smpl_joints_flip_permutation"]:
        cfg_json_dict["constants"]["smpl_pose_flip_permutation"].append(3 * i),
        cfg_json_dict["constants"]["smpl_pose_flip_permutation"].append(3 * i + 1),
        cfg_json_dict["constants"]["smpl_pose_flip_permutation"].append(3 * i + 2),

    cfg_json_dict["constants"]["j49_flip_permutation"] = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18,
                                                          17, 22, 23, 24, 19, 20, 21] + [25 + i for i in
                                                                                         cfg_json_dict["constants"][
                                                                                             "j24_flip_permutation"]],
    cfg_json_dict["constants"]["smpl_j49_flip_permutation"] = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15,
                                                               18, 17, 22, 23, 24, 19, 20, 21] + [25 + i for i in
                                                                                                  cfg_json_dict[
                                                                                                      "constants"][
                                                                                                      "smpl_joints_flip_permutation"]]

    if cfg_json_dict["run"]["is_debug"]:
        cfg_json_dict["train"]["batch_size"] = 4
        cfg_json_dict['train']['save_steps'] = 5
        cfg_json_dict['train']['log_steps'] = 5
        cfg_json_dict['train']['test_steps'] = 5
        pass


    if not osp.exists(osp.join(cfg_json_dict["run"]["output_dir"], "models")):
        os.makedirs(osp.join(cfg_json_dict["run"]["output_dir"], "models"))

    if cfg_json_dict["run"]["platform"] == "Drolab":
        cfg_json_dict["run"]["base_data_dir"] = r"H:\datas\three_dimension_reconstruction\pymaf_family"
        cfg_json_dict["run"]["num_works"] = 0

    elif cfg_json_dict["run"]["platform"] == "dlw":
        cfg_json_dict["run"]["base_data_dir"] = osp.join(r"H:\datas\three_dimension_reconstruction\hybrik")
        cfg_json_dict["run"]["num_works"] = 4

    elif cfg_json_dict["run"]["platform"] == "songbo" and osp.exists(r"/home/ml_group/songbo/danglingwei204"):
        cfg_json_dict["run"]["base_data_dir"] = osp.join(r"/home/ml_group/songbo/danglingwei204/datasets/three_dimension_reconstruction/pymaf_family")
        cfg_json_dict["run"]["num_works"] = 4

    cfg_json_dict["run"]["preprocessed_data_dir"] = osp.join(cfg_json_dict["run"]["base_data_dir"], 'spin_pymaf_data')

    cfg_json_dict["run"]["smpl_mean_params_path"] = osp.join(cfg_json_dict["run"]["preprocessed_data_dir"],
                                                             'smpl_mean_params.npz')
    cfg_json_dict["run"]["smpl_model_path"] = osp.join(cfg_json_dict["run"]["preprocessed_data_dir"], 'smpl')
    cfg_json_dict["run"]["J_regressor_train_extra_path"] = osp.join(cfg_json_dict["run"]["preprocessed_data_dir"],
                                                                    'J_regressor_extra.npy')
    cfg_json_dict["run"]["J_regressor_h36m_path"] = osp.join(cfg_json_dict["run"]["preprocessed_data_dir"],
                                                             'J_regressor_h36m.npy')
    cfg_json_dict["run"]["final_fits_path"] = osp.join(cfg_json_dict["run"]["preprocessed_data_dir"], 'spin_fits')
    cfg_json_dict["run"]["static_fits_path"] = osp.join(cfg_json_dict["run"]["preprocessed_data_dir"], 'static_fits')
    cfg_json_dict["run"]["mesh_down_path"] = osp.join(cfg_json_dict["run"]["preprocessed_data_dir"],
                                                      'mesh_downsampling.npz')
    cfg_json_dict["run"]["uv_data_path"] = osp.join(cfg_json_dict["run"]["preprocessed_data_dir"], "UV_data")

    return cfg_json_dict
