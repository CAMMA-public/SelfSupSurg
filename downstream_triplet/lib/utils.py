# utility functions for CholecT50 based training and eval

import os
import yaml
import time
import random
import numpy as np
from collections import OrderedDict as odict
import torch
import torch.nn.functional as F

from collections import OrderedDict
from easydict import EasyDict as edict

import torch


# get video list
def get_video_list(args):
    split_name = args.TRAIN.SPLIT
    video_split = split_selector(split_name)
    print("video_split, split_name", video_split, split_name)
    train_videos = (
        sum([v for k, v in video_split.items() if k != args.TRAIN.FOLD], [])
        if "cv" in split_name
        else video_split["train"]
    )
    test_videos = (
        sum([v for k, v in video_split.items() if k == args.TRAIN.FOLD], [])
        if "cv" in split_name
        else video_split["test"]
    )

    if "cv" in split_name:
        val_videos = train_videos[-5:]
        train_videos = train_videos[:-5]
    else:
        val_videos = video_split["val"]

    # extra sort similar to that in splits.py
    train_videos = sorted(train_videos)
    val_videos = sorted(val_videos)
    test_videos = sorted(test_videos)

    # in records format
    train_records = ["VID{}".format(str(v).zfill(2)) for v in train_videos]
    val_records = ["VID{}".format(str(v).zfill(2)) for v in val_videos]
    test_records = ["VID{}".format(str(v).zfill(2)) for v in test_videos]

    return train_records, val_records, test_records


def split_selector(case="ctp"):
    # ctp: cholectriplet split
    switcher = {
        "ctp_125_0": {
            "train": [14, 23, 40, 12, 49],
            "val": [5, 18, 36, 65, 74],
            "test": [92, 96, 103, 110, 111],
        },
        "ctp_125_1": {
            "train": [12, 27, 32, 75, 79],
            "val": [5, 18, 36, 65, 74],
            "test": [92, 96, 103, 110, 111],
        },
        "ctp_125_2": {
            "train": [26, 29, 31, 60, 73],
            "val": [5, 18, 36, 65, 74],
            "test": [92, 96, 103, 110, 111],
        },
        "ctp_25_0": {
            "train": [1, 4, 6, 15, 25, 26, 51, 56, 57, 73],
            "val": [5, 18, 36, 65, 74],
            "test": [92, 96, 103, 110, 111],
        },
        "ctp_25_1": {
            "train": [14, 29, 31, 32, 40, 43, 51, 52, 66, 79],
            "val": [5, 18, 36, 65, 74],
            "test": [92, 96, 103, 110, 111],
        },
        "ctp_25_2": {
            "train": [29, 35, 43, 47, 49, 50, 60, 75, 79, 80],
            "val": [5, 18, 36, 65, 74],
            "test": [92, 96, 103, 110, 111],
        },
        "ctp_100_0": {
            "train": [
                1,
                15,
                26,
                40,
                52,
                79,
                2,
                27,
                43,
                56,
                66,
                4,
                22,
                31,
                47,
                57,
                68,
                23,
                35,
                48,
                60,
                70,
                13,
                25,
                49,
                62,
                75,
                8,
                12,
                29,
                50,
                78,
                6,
                51,
                10,
                73,
                14,
                32,
                80,
                42,
            ],
            "val": [5, 18, 36, 65, 74],
            "test": [92, 96, 103, 110, 111],
        },
    }
    return switcher.get(case)


# bce loss for training
def bce_loss(preds, gt, pos_wt=None):
    wt = torch.tensor(pos_wt, device=gt.device) if pos_wt != None else None
    return F.binary_cross_entropy_with_logits(preds, gt, pos_weight=wt)


# display params
def display_model_params(model, show_grad=False):
    for i, j in model.named_parameters():
        if show_grad:
            print(f"{i} >>  {j.requires_grad}")
        else:
            print(i)


# freeze part of the network
def freeze_model(model, exclude_options=["decoder"]):
    count_grad_vars = 0
    for k, v in model.named_parameters():
        if any([j in k for j in exclude_options]):
            count_grad_vars += 1
            continue
        v.requires_grad = False


# load checkpoints
def load_ckpt(model, checkpoint_name):
    checkpoint = torch.load(checkpoint_name)
    if "classy_state_dict" in checkpoint:
        state_dict = checkpoint["classy_state_dict"]["base_model"]["model"]["trunk"]
        state_dict = odict(
            {k.replace("_feature_blocks.", ""): v for k, v in state_dict.items()}
        )
        m, v = model.load_state_dict(state_dict, strict=False)
        print(f"Pretrained model loaded. missing keys: {m}, invalid keys:{v}")
    else:
        model_dict = model.state_dict()
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # to exclude weights not present in ckpt
        state_dict_new = OrderedDict({})

        for k, v in state_dict.items():
            state_dict_new[k] = state_dict[k]

        model_dict.update(state_dict_new)
        m, v = model.load_state_dict(model_dict)
        print(f"Pretrained model loaded. missing keys: {m}, invalid keys:{v}")


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1.0):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def update_acc(self, val, n=1.0):
        self.val = val / n
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def parse_config(config_file):
    with open(config_file) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
    return config


def merge_a_into_b(a, b):
    for k, v in vars(a).items():
        if k not in b:
            b[k] = v


def set_seed(args):
    np.random.seed(args.TRAIN.SEED)
    torch.manual_seed(args.TRAIN.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.TRAIN.SEED)


def check_grad(model):
    for p in model.parameters():
        print("requires_grad >> ", p.requires_grad)


def freeze_batch_norm_2d(module):
    """
    Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`. If `module` is
    itself an instance of either `BatchNorm2d` or `SyncBatchNorm`, it is converted into `FrozenBatchNorm2d` and
    returned. Otherwise, the module is walked recursively and submodules are converted in place.
    Args:
        module (torch.nn.Module): Any PyTorch module.
    Returns:
        torch.nn.Module: Resulting module
    Inspired by https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762
    """
    res = module
    if isinstance(
        module,
        (
            torch.nn.modules.batchnorm.BatchNorm2d,
            torch.nn.modules.batchnorm.SyncBatchNorm,
        ),
    ):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for name, child in module.named_children():
            new_child = freeze_batch_norm_2d(child)
            if new_child is not child:
                res.add_module(name, new_child)
    return res


if __name__ == "__main__":
    pass
