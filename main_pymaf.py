#!/usr/bin/env python
# encoding: utf-8
'''
@project : pymaf_reimp
@file    : main_pymaf.py
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
parser.add_argument('--resume', dest='resume', default=False, action='store_true', help='Resume from checkpoint (Use latest checkpoint by default')
parser.add_argument('--pretrained_checkpoint', default="", help='Load a pretrained checkpoint at the beginning training')
parser.add_argument('--is_single_dataset', default=False, action='store_true', help='Use a single dataset')
parser.add_argument('--is_debug', default=False, action='store_true', help='Use a single dataset')

args = parser.parse_args()

print("\n================== Arguments =================")
pprint(vars(args), indent=4)
print("==========================================\n")

r = RunnerPymaf(exp_name="d211227_pymaf_reimp", is_debug=args.is_debug, args=args)


if args.is_load:
    r.restore(args.model_path)

if args.is_train:
    r.run()
else:
    # r.choose_eval()
    div, ade, mmade, fde, mmfde = r.eval(epoch=-1, draw=False)


