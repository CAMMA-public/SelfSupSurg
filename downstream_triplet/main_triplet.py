"""
Main trainer on triplet dataset
"""

import os
import random
import logging
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from lib.utils import *
from lib.trainer import TripletTrainer

def parse_args():
    """
    Parse input arguments
    """
    # model training details
    parser = argparse.ArgumentParser(description='Train SSL model on CholecTriplet')
    parser.add_argument('--en', default="ssl_triplet_exp1", type=str, help='experiment_name')
    parser.add_argument('--cf', default="./cholec_to_triplet/series_01/100/0/moco.yaml",
                      type=str, help='config yaml file to use for exps')

    # mode - train or eval
    parser.add_argument('--exp_mode', default="train", type=str, choices=['train', 'eval'],
                      help='train or eval')
    parser.add_argument('--ckp_n', default="moco-fc-ft1", type=str,
                      help='experiment name to load model weights')    
    parser.add_argument('--ckp_e', default='', type=str,
                      help='epoch id to load model')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    torch.cuda.empty_cache()

    # parse args and config file
    args_main = parse_args()
    args = parse_config(args_main.cf).CONFIG
    merge_a_into_b(args_main, args)

    # check if the model weights path exists
    if not os.path.exists(os.path.join(args.MODEL.SAVEDIR, args.en)):
        os.makedirs(os.path.join(args.MODEL.SAVEDIR, args.en))
    
    # log file
    args.logfile = os.path.join(args.MODEL.SAVEDIR, args.en, f'{args.en}.log')
    logging.basicConfig(filename=args.logfile, level=logging.INFO, format='%(message)s')

    # seed
    set_seed(args)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    trainer = TripletTrainer(args)

    # train or eval mode ..
    if args.exp_mode == 'train':
        train_loss = trainer.train()
    elif args.exp_mode == 'eval':
        trainer.test()
    else:
        raise ValueError("Incorrect exp mode specified!")