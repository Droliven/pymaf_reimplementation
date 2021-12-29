#!/usr/bin/env python
# encoding: utf-8
'''
@project : pymaf_reimp
@file    : eval_coco_pymaf.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-12-29 15:08
'''
"""
This script can be used to evaluate a trained model on 3D pose/shape and masks/part segmentation. You first need to download the datasets and preprocess them.
Example usage:
```
python3 eval_coco.py --checkpoint=data/pretrained_model/PyMAF_model_checkpoint.pt
```
Running the above command will compute the 2D keypoint detection error. The ```--dataset``` option can take different values based on the type of evaluation you want to perform:
1. COCO ```--dataset=coco```
"""

import os.path as osp
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import pprint

from ..datas import COCODataset
from ..nets import SMPL, PyMAF
from ..utils.geometry import perspective_projection
from ..utils.transforms import transform_preds
from ..utils.uv_vis import vis_smpl_iuv
from ..cfgs import ConfigPymaf


def run_evaluation(model, dataset, result_file,
                   batch_size=32, img_res=224,
                   num_workers=32, shuffle=False):
    """Run evaluation on the datasets and metrics we report in the paper. """
    model.eval()

    device = cfg.device

    # Transfer model to the GPU
    model.to(device)

    # Load SMPL model
    smpl_neutral = SMPL(cfg.SMPL_MODEL_DIR,
                        create_transl=False).to(device)

    save_results = result_file is not None
    # Disable shuffling if you want to save the results
    if save_results:
        shuffle = False
    # Create dataloader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # Store SMPL parameters
    smpl_pose = np.zeros((len(dataset), 72))
    smpl_betas = np.zeros((len(dataset), 10))
    smpl_camera = np.zeros((len(dataset), 3))
    pred_joints = np.zeros((len(dataset), 17, 3))

    num_joints = 17
    num_samples = len(dataset)
    print('dataset length: {}'.format(num_samples))
    all_preds = np.zeros(
        (num_samples, num_joints, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        for _, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
            if len(args.vis_imname) > 0:
                imgnames = [i_n.split('/')[-1] for i_n in batch['imgname']]
                name_hit = False
                for i_n in imgnames:
                    if args.vis_imname in i_n:
                        name_hit = True
                        print('vis: ' + i_n)
                if not name_hit:
                    continue

            images = batch['img'].to(device)

            scale = batch['scale'].numpy()
            center = batch['center'].numpy()

            num_images = images.size(0)

            gt_keypoints_2d = batch['keypoints']  # 2D keypoints
            # De-normalize 2D keypoints from [-1,1] to pixel space
            gt_keypoints_2d_orig = gt_keypoints_2d.clone()
            gt_keypoints_2d_orig[:, :, :-1] = 0.5 * img_res * (gt_keypoints_2d_orig[:, :, :-1] + 1)

            preds_dict, _ = model(images)
            pred_rotmat = preds_dict['smpl_out'][-1]['rotmat'].contiguous().view(-1, 24, 3, 3)
            pred_betas = preds_dict['smpl_out'][-1]['theta'][:, 3:13].contiguous()
            pred_camera = preds_dict['smpl_out'][-1]['theta'][:, :3].contiguous()

            pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                       global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)

            # pred_vertices = pred_output.vertices
            pred_J24 = pred_output.joints[:, -24:]
            pred_JCOCO = pred_J24[:, cfg.J24_TO_JCOCO]

            # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
            # This camera translation can be used in a full perspective projection
            pred_cam_t = torch.stack([pred_camera[:, 1],
                                      pred_camera[:, 2],
                                      2 * cfg.FOCAL_LENGTH / (img_res * pred_camera[:, 0] + 1e-9)], dim=-1)
            camera_center = torch.zeros(len(pred_JCOCO), 2, device=pred_camera.device)
            pred_keypoints_2d = perspective_projection(pred_JCOCO,
                                                       rotation=torch.eye(3, device=pred_camera.device).unsqueeze(
                                                           0).expand(len(pred_JCOCO), -1, -1),
                                                       translation=pred_cam_t,
                                                       focal_length=cfg.FOCAL_LENGTH,
                                                       camera_center=camera_center)

            coords = pred_keypoints_2d + (img_res / 2.)
            coords = coords.cpu().numpy()

            gt_keypoints_coco = gt_keypoints_2d_orig[:, -24:][:, cfg.J24_TO_JCOCO]
            vert_errors_batch = []
            for i, (gt2d, pred2d) in enumerate(zip(gt_keypoints_coco.cpu().numpy(), coords.copy())):
                vert_error = np.sqrt(np.sum((gt2d[:, :2] - pred2d[:, :2]) ** 2, axis=1))
                vert_error *= gt2d[:, 2]
                vert_mean_error = np.sum(vert_error) / np.sum(gt2d[:, 2] > 0)
                vert_errors_batch.append(10 * vert_mean_error)

            preds = coords.copy()

            scale_ = np.array([scale, scale]).transpose()

            # Transform back
            for i in range(coords.shape[0]):
                preds[i] = transform_preds(
                    coords[i], center[i], scale_[i], [img_res, img_res]
                )

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = 1.
            all_boxes[idx:idx + num_images, 5] = 1.
            image_path.extend(batch['imgname'])

            idx += num_images

        if len(args.vis_imname) > 0:
            exit()

        if args.checkpoint is None or 'model_checkpoint.pt' in args.checkpoint:
            ckp_name = 'spin_model'
        else:
            ckp_name = args.checkpoint.split('/')
            ckp_name = ckp_name[2].split('_')[1] + '_' + ckp_name[-1].split('.')[0]
        name_values, perf_indicator = dataset.evaluate(
            cfg, all_preds, args.output_dir, all_boxes, image_path, ckp_name,
            filenames, imgnums
        )

        model_name = args.regressor
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

    # Save reconstructions to a file for further processing
    if save_results:
        np.savez(result_file, pred_joints=pred_joints, pose=smpl_pose, betas=smpl_betas, camera=smpl_camera)


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    print(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    print('|---' * (num_values + 1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    print(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


if __name__ == '__main__':

    # Define command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='coco', help='Choose evaluation dataset')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for testing')
    parser.add_argument('--regressor', type=str, choices=['hmr', 'pymaf_net'], default='pymaf_net',
                        help='Name of the SMPL regressor.')
    parser.add_argument('--result_file', default=None, help='If set, save detections to a .npz file')
    parser.add_argument('--output_dir', type=str, default='./notebooks/output/', help='output directory.')
    parser.add_argument('--vis_demo', default=False, action='store_true', help='result visualization')
    parser.add_argument('--ratio', default=1, type=int, help='image size ration for visualization')
    parser.add_argument('--vis_imname', type=str, default='', help='image name used for visualization.')

    parser.add_argument('--checkpoint', default=r"G:\second_model_report_data\datas\three_dimension_reconstruction\spin_pymaf_data\pretrained_model\PyMAF_model_checkpoint.pt", help='Path to network checkpoint')

    args = parser.parse_args()
    cfg = ConfigPymaf()

    print("\n================== Arguments =================")
    pprint(vars(args), indent=4)
    print("==========================================\n")

    print("\n================== Configs =================")
    pprint(vars(cfg), indent=4)
    print("==========================================\n")

    # 模型
    # PyMAF model
    model = PyMAF(cfg.pymaf_model['BACKBONE'], cfg.res_model['DECONV_WITH_BIAS'],
                  cfg.res_model['NUM_DECONV_LAYERS'], cfg.res_model['NUM_DECONV_FILTERS'],
                  cfg.res_model['NUM_DECONV_KERNELS'], cfg.pymaf_model['MLP_DIM'],
                  cfg.pymaf_model['N_ITER'], cfg.pymaf_model['AUX_SUPV_ON'], cfg.BN_MOMENTUM,
                  cfg.SMPL_MODEL_DIR, cfg.H36M_TO_J14, cfg.LOSS['POINT_REGRESSION_WEIGHTS'],
                  JOINT_MAP=cfg.JOINT_MAP, JOINT_NAMES=cfg.JOINT_NAMES, J24_TO_J19=cfg.J24_TO_J19,
                  JOINT_REGRESSOR_TRAIN_EXTRA=cfg.JOINT_REGRESSOR_TRAIN_EXTRA,
                  device=cfg.device, SMPL_MEAN_PARAMS_PATH=cfg.SMPL_MEAN_PARAMS_PATH, pretrained=True,
                  data_dir=cfg.preprocessed_data_dir)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'], strict=True)
        print(f"loaded from {args.checkpoint}!")


    # Setup evaluation dataset
    dataset = COCODataset(eval_pve=cfg.eval_pve, noise_factor=cfg.noise_factor, rot_factor=cfg.rot_factor, scale_factor=cfg.scale_factor, ds=args.dataset, subset="val2014", ignore_3d=False, use_augmentation=True, is_train=False, DATASET_FOLDERS=cfg.ORIGIN_IMGS_DATASET_FOLDERS, DATASET_FILES=cfg.PREPROCESSED_DATASET_FILES, JOINT_MAP=cfg.JOINT_MAP, JOINT_NAMES=cfg.JOINT_NAMES, J24_TO_J19=cfg.J24_TO_J19, JOINT_REGRESSOR_TRAIN_EXTRA=cfg.JOINT_REGRESSOR_TRAIN_EXTRA, SMPL_MODEL_DIR=cfg.SMPL_MODEL_DIR, IMG_NORM_MEAN=cfg.IMG_NORM_MEAN, IMG_NORM_STD=cfg.IMG_NORM_STD, TRAIN_BATCH_SIZE=cfg.TRAIN_BATCHSIZE, IMG_RES=cfg.IMG_RES, SMPL_JOINTS_FLIP_PERM=cfg.SMPL_JOINTS_FLIP_PERM)

    # Run evaluation
    args.result_file = None
    run_evaluation(model, dataset, args.result_file, batch_size=args.batch_size, shuffle=False, num_workers=cfg.num_works)

    print('{}: {}, {}'.format(args.regressor, args.checkpoint, args.dataset))