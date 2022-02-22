#!/usr/bin/env python
# encoding: utf-8
'''
@project : pymaf_reimp
@file    : main_d211227_pymaf.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-12-27 15:52
'''

# ****************************************************************************************************************
# *********************************************** Environments ***************************************************
# ****************************************************************************************************************

import numpy as np
import random
import torch
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def seed_torch(seed=3450):
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

seed_torch()


# ****************************************************************************************************************
# *********************************************** Main ***********************************************************
# ****************************************************************************************************************

import argparse
import json

from d211227_pymaf_reimp.runs import RunPymafReimp

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--is_load', default='1', type=bool)
parser.add_argument('--is_single_dataset', default='', type=bool)
parser.add_argument('--is_debug', default='', type=bool)
parser.add_argument('--is_train', default='', type=bool)

parser.add_argument('--pretrained_checkpoint_path', default=r"G:\second_model_report_data\report_hmr\pymaf_reimp\alldata_single21_mix40\results\d211227_pymaf_reimp_single\models\model_epoch_00000021.pt", help='Load a pretrained checkpoint at the beginning training')
# parser.add_argument('--pretrained_checkpoint_path', default=r"H:\datas\three_dimension_reconstruction\pymaf_family\spin_pymaf_data\pretrained_model\PyMAF_model_checkpoint.pt", help='Load a pretrained checkpoint at the beginning training')
# parser.add_argument('--pretrained_checkpoint_path', default=r"/home/ml_group/songbo/danglingwei204/datasets/three_dimension_reconstruction/pymaf_family/spin_pymaf_data/pretrained_model/PyMAF_model_checkpoint.pt", help='Load a pretrained checkpoint at the beginning training')

args = parser.parse_args()

print("\n================== Configs =================")
print(json.dumps(args.__dict__, indent=4, ensure_ascii=False, separators=(", ", ": ")))
print("==========================================\n")

r = RunPymafReimp(exp_name="d211227_pymaf_reimp", is_debug=args.is_debug, args=args)


if args.is_load:
    r.load_checkpoint(args.pretrained_checkpoint_path)

if args.is_train:
    r.run()
else:
    mpjpe, pampjpe, pve = r.test()
    print(
        f'Test {r.valid_loader.dataset.dataset} || MPJPE: {mpjpe.mean():.4f}, PAMPJPE: {pampjpe.mean():.4f}, PVE: {pve.mean():.4f}')


