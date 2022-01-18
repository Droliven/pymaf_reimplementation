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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
import pandas as pd
from pprint import pprint

from d211227_pymaf_reimp.runs import RunnerPymaf

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--is_load', default='', type=bool)
parser.add_argument('--is_single_dataset', default='', type=bool)
parser.add_argument('--is_debug', default='1', type=bool)
parser.add_argument('--is_train', default='1', type=bool)

parser.add_argument('--pretrained_checkpoint_path', default="", help='Load a pretrained checkpoint at the beginning training')

args = parser.parse_args()

print("\n================== Arguments =================")
pprint(vars(args), indent=4)
print("==========================================\n")

r = RunnerPymaf(exp_name="d211227_pymaf_reimp", is_debug=args.is_debug, args=args)


if args.is_load:
    r.load_checkpoint(args.pretrained_checkpoint_path)

if args.is_train:
    r.run()
else:
    pass


