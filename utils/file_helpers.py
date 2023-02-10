'''
Project: SelfSupSurg
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''

from glob import glob
import os
import yaml
from types import SimpleNamespace
import json
import pickle
import openpyxl as oxl
from openpyxl.utils.dataframe import dataframe_to_rows
import pandas as pd
import re
import argparse
from easydict import EasyDict as edict
import sys

def parse_hp(hpfile, raw=False):
    with open(hpfile) as hp_raw:
        hp = yaml.safe_load(hp_raw)
    if not raw:
        hp = json.loads(json.dumps(hp), object_hook=lambda d: SimpleNamespace(**d))
    return hp


def ld_txt(txtfile):
    with open(txtfile) as f:
        return [lin.strip() for lin in f.readlines()]


def nm(filename):
    return os.path.basename(filename).split(".")[0]


def mk_tag(*args):
    tokens = [str(a) for a in args if a is not None]
    return "_".join(tokens)


def get_filelist(*args, ratio=None, resample=None, is_train=False, fps=1):
    if is_train:
        return os.path.join("./datasets", *[str(a) for a in args]) + ".pickle"
    else:
        return os.path.join("./datasets", *[str(a) for a in args], str(fps)+"fps.pickle")
    


def get_ds_name(nm):
    return nm


def get_post_ablation_params(ssl):
    ssl_file = os.path.join("xp_prep", "presets", "post_ablation", ssl + ".yaml")
    return parse_hp(ssl_file, raw=True)["config"]


def get_augmentations(*args):
    return {k: v for a in args for k, v in aug(a).items()}


def aug(nm):
    augs = {
        "c1": "xp_prep/presets/aug_sets/color/color_mild.yaml",
        "c2": "xp_prep/presets/aug_sets/color/color_strong.yaml",
        "g1": "xp_prep/presets/aug_sets/geom/geom_mild.yaml",
        "g2": "xp_prep/presets/aug_sets/geom/geom_strong.yaml",
        "barlow": "xp_prep/presets/aug_sets/best/barlow.yaml",
        "moco": "xp_prep/presets/aug_sets/best/moco.yaml",
        "simclr": "xp_prep/presets/aug_sets/best/simclr.yaml",
        "swav": "xp_prep/presets/aug_sets/best/swav.yaml",
        "barlow_default": "xp_prep/presets/aug_sets/defaults/barlow.yaml",
        "moco_default": "xp_prep/presets/aug_sets/defaults/moco.yaml",
        "simclr_default": "xp_prep/presets/aug_sets/defaults/simclr.yaml",
        "swav_default": "xp_prep/presets/aug_sets/defaults/swav.yaml",
    }
    augs = {k: parse_hp(v, raw=True) for k, v in augs.items()}
    augs[""] = {}
    return augs[nm]


def get_arch(nm):
    return nm.split("_")[0].lower()


def get_depth(nm):
    return nm.split("_")[1]


def get_width(nm):
    if "width" in nm:
        return int(nm.split("width_")[-1])
    else:
        return 1

def get_class_weights(path, task, is_weighted=True):
    key = 'phase_mfb' if task == 'phase' else 'tool_ib'
    weights = None
    if is_weighted:
        with open(path, "rb") as fp:
            data = pickle.load(fp)
        weights = data[key]
    return weights

def get_n_tool(dataset):
    return {"cholec80": 7, "cataracts": 21}[dataset]


def get_n_phase(dataset):
    return {"cholec80": 7, "cataracts": 19}[dataset]


def get_loss(ssl_method):
    loss_dict = {
        "barlow": "barlow_twins_loss",
        "moco": "moco_loss",
        "simclr": "simclr_info_nce_loss",
        "swav": "swav_loss",
    }
    return loss_dict[ssl_method]


def get_collate_function(ssl_method):
    collate_function = {
        "barlow": "simclr_collator",
        "moco": "moco_collator",
        "simclr": "simclr_collator",
        "swav": "multicrop_collator",
    }
    return collate_function[ssl_method]


def get_loss_params(ssl_method):
    param_dict = {
        "barlow": {
            "barlow_twins_loss": {
                "lambda_": 0.0051,
                "scale_loss": 0.024,
                "embedding_dim": 128,
            }
        },
        "moco": {
            "moco_loss": {
                "embedding_dim": 128,
                "queue_size": 65536,
                "momentum": 0.999,
                "temperature": 0.2,
            }
        },
        "simclr": {
            "simclr_info_nce_loss": {
                "temperature": 0.1,
                "buffer_params": {"embedding_dim": 128},
            }
        },
        "swav": {
            "swav_loss": {
                "temperature": 0.1,
                "use_double_precision": False,
                "normalize_last_layer": True,
                "num_iters": 3,
                "epsilon": 0.05,
                "crops_for_assign": [0, 1],
                "queue": {"queue_length": 0, "start_iter": 0},
            }
        },
    }
    return param_dict[ssl_method]


def mk_transform(trfms, **kwargs):
    mc, color, geom, severe = trfms
    mc = int(re.sub("[^0-9]", "", mc))
    color = int(re.sub("[^0-9]", "", color))
    geom = int(re.sub("[^0-9]", "", geom))
    severe = int(re.sub("[^0-9]", "", severe))
    d = {
        **kwargs,
        **{"default": 0, "mc": mc, "color": color, "geom": geom, "severe": severe},
    }
    transform_set = {
        k: parse_hp("xp_prep/presets/aug_sets/{}/{}.yaml".format(k, v), raw=True)[
            "transforms"
        ]
        for k, v in d.items()
    }
    transform_merged_pre, transform_merged_post = {}, {}
    augs = sum(
        [
            d["RandAugmentSurgery"]["transforms"]
            for d in transform_set.values()
            if d is not None and "RandAugmentSurgery" in d
        ],
        [],
    )
    for d in transform_set.values():
        if d is not None:
            for k, v in d.items():
                transform_merged_pre[k] = v
    if "RandAugmentSurgery" in transform_merged_pre:
        transform_merged_pre["RandAugmentSurgery"]["transforms"] = augs

    # TODO: Kind of an hack. Make it proper later
    if "RandomErasing" in transform_merged_pre:
        transform_merged_post["RandomErasing"] = transform_merged_pre.pop(
            "RandomErasing"
        )

    transforms = {"pre": transform_merged_pre, "post": transform_merged_post}

    return transforms


def get_downstream_params(dws_method):
    param_dict = {"tcn": {}, "opera": {}, "linear": {}}
    return param_dict[dws_method]


def get_run_dir(run_name, run_id, test_args=[]):
    tokens = [str(a) for a in test_args if a is not None]
    return os.path.join(
        "runs", run_name, "run_{:03d}".format(run_id), "test" if tokens else "", *tokens
    )


def get_extraction_dir(run_name):
    return os.path.join("data", "features", run_name)


def mk_xls(core, defaults, nm, tags):
    df = pd.DataFrame([{**tag, **cv, **defaults} for tag, cv in zip(tags, core)])
    df.index += 1
    wb = oxl.Workbook()
    ws = wb.active
    for r in dataframe_to_rows(df, index=True, header=True):
        ws.append(r)
    wb.save(nm)


def mk_csv(core, defaults, nm, tags):
    df = pd.DataFrame([{**tag, **cv, **defaults} for tag, cv in zip(tags, core)])
    df.index += 1
    df.to_csv(nm)


def get_weights_init_for_epoch(
    run_name, epoch, ssl_method="", test_args=[], is_final=False
):
    tokens = [str(a) for a in test_args if a is not None]
    ckpt_name = (
        "model_phase" + str(epoch) + ".torch"
        if not is_final
        else "model_final_checkpoint_phase" + str(epoch - 1) + ".torch"
    )
    # hard coded for augmentation setting mc8_c1_g1_s0
    if ssl_method == "moco":
        run_id = 1
    elif ssl_method == "simclr":
        run_id = 17
    elif ssl_method == "swav":
        run_id = 33
    elif ssl_method == "dino":
        run_id = 49
    else:
        assert False, "method doesn't exist"

    ckpt_file_name = os.path.join(
        "runs",
        run_name,
        "run_{:03d}".format(run_id),
        ckpt_name,
        "test" if tokens else "",
        *tokens
    )
    if ckpt_file_name:
        ckpt_file_name = ckpt_file_name[0:-1]

    return ckpt_file_name

def get_imagenet_weights(method):
    files = {
        "barlow": "./checkpoints/defaults/resnet_50/barlow_twins_32gpus_4node_imagenet1k_300ep_resnet50.torch",
        # "moco": "./checkpoints/defaults/resnet_50/moco_v2_1node_lr.03_step_b32_zero_init.torch",
        # "simclr": "./checkpoints/defaults/resnet_50/simclr_rn50_200ep_simclr_8node_resnet_16_07_20.a816c0ef.torch",
        # "swav": "./checkpoints/defaults/resnet_50/swav_in1k_rn50_200ep_swav_8node_resnet_27_07_20.bd595bb0.torch",
        # "dino": "./checkpoints/defaults/resnet_50/swav_in1k_rn50_200ep_swav_8node_resnet_27_07_20.bd595bb0.torch",
        "moco": "./checkpoints/defaults/resnet_50/moco_v2_800ep_pretrain.pth.tar",
        "simclr": "./checkpoints/defaults/resnet_50/simclr_rn50_800ep_simclr_8node_resnet_16_07_20.7e8feed1.torch",
        "swav": "./checkpoints/defaults/resnet_50/swav_in1k_rn50_800ep_swav_8node_resnet_27_07_20.a0a6b676.torch",
        "dino": "./checkpoints/defaults/resnet_50/dino_resnet50_pretrain.pth",

        "supervised": "./checkpoints/defaults/resnet_50/resnet50-19c8e357.pth",
    }
    return files[method]



def get_weights_init(provided_dir, arch, custom, method, init_weights_train):
    if custom is not None:
        target_dir = provided_dir
        # WARNING: MAKE SURE THIS PATTERN IS CORRECT
    else:
        if init_weights_train == "imagenet":
            files = {
                "barlow": "barlow_twins_32gpus_4node_imagenet1k_300ep_resnet50.torch",
                "moco": "moco_v2_1node_lr.03_step_b32_zero_init.torch",
                "simclr": "simclr_rn50_200ep_simclr_8node_resnet_16_07_20.a816c0ef.torch",
                "swav": "swav_in1k_rn50_200ep_swav_8node_resnet_27_07_20.bd595bb0.torch",
            }
            target_dir = os.path.join(
                "./checkpoints", "defaults", "{}".format(arch), files[method]
            )
        else:
            target_dir = ""
    return target_dir


def get_final_model_ckpt(path):
    weights_init_file = ""
    if path == "":
        return weights_init_file
    if ".pth" in path or ".torch" in path:
        weights_init_file = path
    else:
        assert os.path.isdir(path), "The folder specified doesn't exists!!!: " + path
        files = sorted(glob(os.path.join(path, "model_final*.*")))
        assert files != [], "No model final weights file found!!!: " + path
        weights_init_file = files[-1]
    assert os.path.isfile(weights_init_file), (
        "No model final weights file found!!!: " + path
    )
    return weights_init_file


def hp_to_slurm(hp, hp_root, slurm_root, provider):
    tail = os.path.basename(hp).split(".")[0]
    tail_out = "s{}.sh".format(tail[1:])
    slurm_path = os.path.join(
        slurm_root, provider, os.path.dirname(os.path.relpath(hp, hp_root)), tail_out
    )
    return slurm_path

def parse_config(config_file):
    with open(config_file) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
    return config

# Argument parser
def create_argument_parser():
    parser = argparse.ArgumentParser(description='Parse model training options')
    parser.add_argument('-hp', '--hyper_params', default='hparams/hp_225_bypass.yaml',
                    help='Path to the hyper parameters config file.\n')
    parser.add_argument('-t', '--test_set', default='val',
                    help='specify the dataset on which to test: test/val\n')
    return parser
