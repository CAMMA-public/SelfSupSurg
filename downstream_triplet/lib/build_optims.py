import os
import time
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim

def init_optims(args, model):
    params = get_model_params(args, model)
    # print("params are ")
    # print(params)

    optimizer, scheduler = None, None
    if args.TRAIN.OPT == 'adam':
        optimizer = optim.Adam(
                        params,
                        lr=hparams.param_schedulers.lr.base_lr,
                    )
    elif args.TRAIN.OPT == 'adamw':
        optimizer = optim.AdamW(
                        params,
                        lr=hparams.param_schedulers.lr.base_lr,
                    )
    elif args.TRAIN.OPT == 'sgd':        
        optimizer = optim.SGD(
                        params,
                        lr=args.TRAIN.LR,
                        momentum=args.TRAIN.MOMENTUM,
                        weight_decay=args.TRAIN.WD1,
                    )
    else:
        raise ValueError("Incorrect optimizer type specified!")

    if args.TRAIN.SCHEDULER == 'multi-step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)
    elif args.TRAIN.SCHEDULER == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.TRAIN.GM1)
    else:
        raise ValueError("Incorrect scheduler type provided!")
            
    return optimizer, scheduler

def get_model_params(args, model):          
    if args.TRAIN.FINETUNE:
        params1, params2 = [], []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if 'backbone' in key:
                    params1 += [{'params':[value], 'lr': args.TRAIN.BACKLR}]
                elif 'fc' in key:
                    params2 += [{'params':[value], 'lr':args.TRAIN.FCLR}]
                else:
                    print(f"Missed {key}")
        param_dict = params1 + params2
    elif args.TRAIN.LINEARPROB:
        #fc_tune = model.down.parameters()
        params2 = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if 'fc' in key:
                    params2 += [{'params':[value], 'lr':args.TRAIN.FCLR}]
                else:
                    print(f"Missed {key}")
        param_dict = params2

    return param_dict