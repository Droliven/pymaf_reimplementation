#!/usr/bin/env python
# encoding: utf-8
'''
@project : pymaf_reimp
@file    : config_pymaf.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-12-27 16:00
'''
import os.path as osp
import getpass
import os

class ConfigPymaf():
    def __init__(self, exp_name=""):
        self.exp_name = exp_name
        self.platform = getpass.getuser()

        self.device = "cuda:0"

        self.openpose_train_weight=0. # 'Weight for OpenPose keypoints during training')
        self.gt_train_weight=1. # 'Weight for GT keypoints during training')

        self.VAL_VIS_BATCH_FREQ = 200
        self.TRAIN_VIS_ITER_FERQ = 1000
        self.TRAIN_VAL_LOOP = True
        self.TRAIN_BATCHSIZE = 64
        self.TEST_BATCHSIZE = 32

        self.SEED_VALUE = -1

        self.epoch_num = 200

        self.SOLVER = {
            'MAX_ITER': 500000,
            'TYPE': 'Adam',
            'BASE_LR': 0.00005,
            'GAMMA': 0.1,
            'STEPS': [0],
            'EPOCHS': [0],
        }

        self.LOSS = {
            'KP_2D_W' : 300.0,
            'KP_3D_W' : 300.0,
            'SHAPE_W' : 0.06,
            'POSE_W' : 60.0,
            'VERT_W' : 0.0,

            # Loss weights for dense correspondences
            'INDEX_WEIGHTS' : 2.0,
            # Loss weights for surface parts. (24 Parts)
            'PART_WEIGHTS' : 0.3,
            # Loss weights for UV regression.
            'POINT_REGRESSION_WEIGHTS' : 0.5,
        }

        self.BN_MOMENTUM = 0.1

        self.pymaf_model = {
            'BACKBONE': 'res50',
            'MLP_DIM': [256, 128, 64, 5],
            'N_ITER': 3,
            'AUX_SUPV_ON': True,
            'DP_HEATMAP_SIZE': 56,
        }

        self.res_model = {
            'DECONV_WITH_BIAS': False,
            'NUM_DECONV_LAYERS': 3,
            'NUM_DECONV_FILTERS': [256, 256, 256],
            'NUM_DECONV_KERNELS': [4, 4, 4],
        }

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> constants <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/constants.py
        self.FOCAL_LENGTH = 5000.
        self.IMG_RES = 224

        # Mean and standard deviation for normalizing input image
        self.IMG_NORM_MEAN = [0.485, 0.456, 0.406]
        self.IMG_NORM_STD = [0.229, 0.224, 0.225]

        """
        We create a superset of joints containing the OpenPose joints together with the ones that each dataset provides.
        We keep a superset of 24 joints such that we include all joints from every dataset.
        If a dataset doesn't provide annotations for a specific joint, we simply ignore it.
        The joints used here are the following:
        """
        self.JOINT_NAMES = [
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
        ]

        # Dict containing the joints in numerical order
        self.JOINT_IDS = {self.JOINT_NAMES[i]: i for i in range(len(self.JOINT_NAMES))}

        # Map joints to SMPL joints
        self.JOINT_MAP = {
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
        }

        # Joint selectors
        # Indices to get the 14 LSP joints from the 17 H36M joints
        self.H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
        self.H36M_TO_J14 = self.H36M_TO_J17[:14]
        # Indices to get the 14 LSP joints from the ground truth joints
        self.J24_TO_J17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17]
        self.J24_TO_J14 = self.J24_TO_J17[:14]
        self.J24_TO_J19 = self.J24_TO_J17[:14] + [19, 20, 21, 22, 23]
        self.J24_TO_JCOCO = [19, 20, 21, 22, 23, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]

        # Permutation of SMPL pose parameters when flipping the shape
        self.SMPL_JOINTS_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
        self.SMPL_POSE_FLIP_PERM = []
        for i in self.SMPL_JOINTS_FLIP_PERM:
            self.SMPL_POSE_FLIP_PERM.append(3 * i)
            self.SMPL_POSE_FLIP_PERM.append(3 * i + 1)
            self.SMPL_POSE_FLIP_PERM.append(3 * i + 2)
        # Permutation indices for the 24 ground truth joints
        self.J24_FLIP_PERM = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18, 19, 21, 20, 23, 22]
        # Permutation indices for the full set of 49 joints
        self.J49_FLIP_PERM = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21] \
                        + [25 + i for i in self.J24_FLIP_PERM]
        self.SMPL_J49_FLIP_PERM = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21] \
                             + [25 + i for i in self.SMPL_JOINTS_FLIP_PERM]

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> path configs <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        self.output_dir = osp.join("./results/", exp_name)
        if not osp.exists(osp.join(self.output_dir, "models")):
            os.makedirs(osp.join(self.output_dir, "models"))

        if self.platform == "Drolab":
            DATASETS_BASE_DIR = r"H:\datas\three_dimension_reconstruction"
            self.num_works = 0
        elif self.platform == "dlw":
            DATASETS_BASE_DIR = r"G:\second_model_report_data\datas\three_dimension_reconstruction"
            self.num_works = 4

        self.preprocessed_data_dir = osp.join(DATASETS_BASE_DIR, 'spin_pymaf_data')
        self.SMPL_MEAN_PARAMS_PATH = osp.join(self.preprocessed_data_dir, 'smpl_mean_params.npz')
        self.SMPL_MODEL_DIR = osp.join(self.preprocessed_data_dir, 'smpl')
        self.JOINT_REGRESSOR_TRAIN_EXTRA = osp.join(self.preprocessed_data_dir, "J_regressor_extra.npy")
        self.JOINT_REGRESSOR_H36M = osp.join(self.preprocessed_data_dir, "J_regressor_h36m.npy")
        self.FINAL_FITS_DIR = osp.join(self.preprocessed_data_dir, "spin_fits")
        self.STATIC_FITS_DIR = osp.join(self.preprocessed_data_dir, "static_fits")
        self.UV_data_path = osp.join(self.preprocessed_data_dir, "UV_data")

        H36M_ROOT = osp.join(DATASETS_BASE_DIR, 'h36m/imgs')  # img/, annotation/
        LSP_ROOT = osp.join(DATASETS_BASE_DIR, 'lsp/lsp_dataset_small')  # img/, annotation/
        LSP_ORIGINAL_ROOT = osp.join(DATASETS_BASE_DIR, 'lsp/lsp_dataset_original') # img/, annotation/
        LSPET_ROOT = osp.join(DATASETS_BASE_DIR, 'lsp/hr_lspet') # img/
        MPII_ROOT = osp.join(DATASETS_BASE_DIR, 'mpii') # imgs/
        COCO_ROOT = osp.join(DATASETS_BASE_DIR, 'coco') # imgs/, annotations/
        PW3D_ROOT = osp.join(DATASETS_BASE_DIR, '3dpw') # imgs
        MPI_INF_3DHP_ROOT = osp.join(DATASETS_BASE_DIR, 'mpi_inf_3dhp/mpi_inf_3dhp_train_set')
        UPI_S1H_ROOT = osp.join(DATASETS_BASE_DIR, 'upi_s1h')
        SURREAL_ROOT = osp.join(DATASETS_BASE_DIR, 'SURREAL/data')
        threeDOH50K_ROOT = osp.join(DATASETS_BASE_DIR, '3DOH50K')
        self.ORIGIN_IMGS_DATASET_FOLDERS = {
                    'h36m': H36M_ROOT,
                    'h36m_p1': H36M_ROOT,
                    'h36m_p2': H36M_ROOT,
                    'h36m_p2_mosh': H36M_ROOT,

                    'lsp_orig': LSP_ORIGINAL_ROOT,
                    'lsp': LSP_ROOT,
                    'lspet': LSPET_ROOT,
                    'mpii': MPII_ROOT,
                    'coco': COCO_ROOT,
                    # 'dp_coco': COCO_ROOT,
                    'mpi_inf_3dhp': MPI_INF_3DHP_ROOT,
                    '3dpw': PW3D_ROOT,
                    # 'upi_s1h': UPI_S1H_ROOT,
                    # 'surreal': SURREAL_ROOT,
                    # '3doh50k': threeDOH50K_ROOT
                }

        self.PREPROCESSED_DATASET_FILES = [{
                    'h36m_p1': osp.join(self.preprocessed_data_dir, "dataset_extras", 'h36m_valid_protocol1.npz'),
                    'h36m_p2': osp.join(self.preprocessed_data_dir, "dataset_extras", 'h36m_valid_protocol2.npz'),
                    'h36m_p2_mosh': osp.join(self.preprocessed_data_dir, "dataset_extras", 'h36m_mosh_valid_p2.npz'),
                    'lsp': osp.join(self.preprocessed_data_dir, "dataset_extras", 'lsp_dataset_test.npz'),
                    # 'mpi_inf_3dhp': osp.join(self.preprocessed_data_dir, "dataset_extras", 'mpi_inf_3dhp_test.npz'), # 这个没有
                    'mpi_inf_3dhp': osp.join(self.preprocessed_data_dir, "dataset_extras", 'mpi_inf_3dhp_valid.npz'),
                    '3dpw': osp.join(self.preprocessed_data_dir, "dataset_extras", '3dpw_test.npz'),
                    'coco': osp.join(self.preprocessed_data_dir, "dataset_extras", 'coco_2014_val.npz'),
                    # 'dp_coco': osp.join(self.preprocessed_data_dir, "dataset_extras", 'dp_coco_2014_minival.npz'), # 这个没有
                    # 'surreal': osp.join(self.preprocessed_data_dir, "dataset_extras", 'surreal_val.npz'), # 这个没有
                    # '3doh50k': osp.join(self.preprocessed_data_dir, "dataset_extras", 'threeDOH50K_testset.npz')  # 这个没有
                },
                {
                    'h36m': osp.join(self.preprocessed_data_dir, "dataset_extras", 'h36m_mosh_train.npz'),
                    'lsp_orig': osp.join(self.preprocessed_data_dir, "dataset_extras", 'lsp_dataset_original_train.npz'),
                    'lspet': osp.join(self.preprocessed_data_dir, "dataset_extras", 'hr_lspet_train.npz'),
                    'mpii': osp.join(self.preprocessed_data_dir, "dataset_extras", 'mpii_train.npz'),
                    'coco': osp.join(self.preprocessed_data_dir, "dataset_extras", 'coco_2014_train.npz'),
                    # 'dp_coco': osp.join(self.preprocessed_data_dir, "dataset_extras", 'dp_coco_2014_train.npz'),  # 这个没有
                    'mpi_inf_3dhp': osp.join(self.preprocessed_data_dir, "dataset_extras", 'mpi_inf_3dhp_train.npz'),
                    # 'surreal': osp.join(self.preprocessed_data_dir, "dataset_extras", 'surreal_train.npz'),  # 这个没有
                    # '3doh50k': osp.join(self.preprocessed_data_dir, "dataset_extras", 'threeDOH50K_trainset.npz') # 这个没有
                }]

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> train options <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        self.single_dataname = "h36m"
        self.eval_dataset = "h36m_p2_mosh"
        self.eval_pve = False
        self.noise_factor = 0.4
        self.rot_factor = 30
        self.scale_factor = 0.25
