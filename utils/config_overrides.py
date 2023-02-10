'''
Project: SelfSupSurg
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''

import os
import glob
import yaml
import argparse
import utils.file_helpers as fh


def ssl_overrides(config_file, split=None, feature=""):
    # TODO: verify overrides
    config = yaml.safe_load(open(os.path.join("./configs/config", config_file), "r"))
    weights_init_path = config["config"]["MODEL"]["WEIGHTS_INIT"]["PARAMS_FILE"]

    overrides = [
        "config=" + config_file,
        #'config.MODEL.WEIGHTS_INIT.PARAMS_FILE=' + fh.get_final_model_ckpt(weights_init_path),
    ]

    return overrides


def supervised_overrides(config_file, split=None, feature=""):
    config = yaml.safe_load(open(os.path.join("./configs/config", config_file), "r"))
    weights_init_path = config["config"]["MODEL"]["WEIGHTS_INIT"]["PARAMS_FILE"]

    overrides = [
        "config=" + config_file,
        "config.MODEL.WEIGHTS_INIT.PARAMS_FILE="
        + fh.get_final_model_ckpt(weights_init_path),
    ]
    return overrides


def extract_features_overrides(config_file, split="train", feature=""):
    # TODO: add list of overrides, engine: extract_features, get the best
    # checkpoint path of the ssl/supervised model, trunk or head feature extraction
    config = yaml.safe_load(open(os.path.join("./configs/config", config_file), "r"))

    feature_extract = config["config"]["FEATURE_EXTRACT"]
    weights_init_path = config["config"]["CHECKPOINT"]["DIR"]
    ishead = feature_extract == "Head"
    if ishead and feature == "Trunk": ishead = False
    test_only = ishead
    # if 'main_im2' in config["config"]["TAG"]: weights_init_path = config["config"]["MODEL"]["WEIGHTS_INIT"]["PARAMS_FILE"]
    if ishead:
        ckpt_path = os.path.join(config["config"]["RUN_DIR"], "extracted_features")
        # assert not os.path.exists(ckpt_path), 'feature extraction path already exists!!!: ' + ckpt_path
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        weights_init_file = fh.get_final_model_ckpt(weights_init_path)
        overrides = [
            "engine_name=extract_features",
            "hydra.verbose=true",
            "config=" + config_file,
            "config.CHECKPOINT.DIR=" + ckpt_path,
            "config.TEST_ONLY=" + str(test_only),
            "config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_TRUNK_AND_HEAD=" + str(ishead),
            "config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY="
            + str(not ishead),
            "config.MODEL.WEIGHTS_INIT.PARAMS_FILE="
            + weights_init_file,
        ]
    else:
        ckpt_path = os.path.join(
            config["config"]["RUN_DIR"], "extracted_features_" + feature_extract, split
        )
        weights_init_file = fh.get_final_model_ckpt(weights_init_path)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        # replace("120", "80") is for cholec120 pre-training (we need the feature extraction for cholec80)
        overrides = [
            "engine_name=extract_features",
            "hydra.verbose=true",
            "config=" + config_file,
            "config.CHECKPOINT.DIR=" + ckpt_path,
            "config.FEATURE_EXTRACTION=True",
            "config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY=True",
            "config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON=True",
            "config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS=True",
            "config.DATA.TRAIN.DROP_LAST=False",
            "config.DATA.TEST.DROP_LAST=False",
            "config.DATA.VAL.DROP_LAST=False",
            "config.DATA.TRAIN.DATA_PATHS="+str(config["config"]["DATA"]["TRAIN"]["DATA_PATHS"]).replace("p33fps", "1fps").replace("p5fps", "1fps").replace("5fps", "1fps").replace("3fps", "1fps").replace("p1fps", "1fps").replace("120", "80"),
            "config.DATA.TRAIN.LABEL_PATHS="+str(config["config"]["DATA"]["TRAIN"]["DATA_PATHS"]).replace("p33fps", "1fps").replace("p5fps", "1fps").replace("5fps", "1fps").replace("3fps", "1fps").replace("p1fps", "1fps").replace("120", "80"),
            "config.DATA.TRAIN.LABEL_SOURCES="+str(config["config"]["DATA"]["TRAIN"]["DATA_SOURCES"]),
            "config.DISTRIBUTED.NUM_PROC_PER_NODE=4",
	        "config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=256",
            "config.DATA.TEST.BATCHSIZE_PER_REPLICA=256",
            "config.DATA.VAL.BATCHSIZE_PER_REPLICA=256",
            "config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=classy_state_dict",
            "config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_TRUNK_AND_HEAD=" + str(ishead),
            "config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY="
            + str(not ishead),
            "config.MODEL.WEIGHTS_INIT.PARAMS_FILE="
            + weights_init_file,
        ]
        if split == "train":
            overrides.append("config.TEST_ONLY=False")
            overrides.append("config.TEST_MODEL=False")
        elif split == "test" or split == "val":
            overrides.append("config.TEST_ONLY=True")
            overrides.append("config.TEST_MODEL=True")
            if split == "val":
                data_paths = config["config"]["DATA"]["VAL"]["DATA_PATHS"]
                label_paths = config["config"]["DATA"]["VAL"]["LABEL_PATHS"]
                overrides.append("config.DATA.TEST.DATA_PATHS=" + str(data_paths))
                overrides.append("config.DATA.TEST.LABEL_PATHS=" + str(label_paths))
        else:
            assert False, "unknown split"

    if '.torch' in weights_init_file:
        overrides += [
            "config.MODEL.WEIGHTS_INIT.APPEND_PREFIX=",
            "config.MODEL.WEIGHTS_INIT.REMOVE_PREFIX=",
            "config.MODEL.WEIGHTS_INIT.SKIP_LAYERS=",
            "config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=classy_state_dict",
        ]

    return overrides


def create_argument_parser():
    parser = argparse.ArgumentParser(description="Parse model training options")
    parser.add_argument(
        "-m",
        "--mode",
        default="self_supervised",
        help="State of the model.\n      Options: self_supervised, supervised, feature_extraction",
    )

    parser.add_argument(
        "-s",
        "--split",
        default="train",
        required=False,
        help="split for the feature extraction.\n      Options: train, test, val",
    )

    parser.add_argument(
        "-f",
        "--feature",
        default="",
        help="State of the model.\n      Options: Trunk",
    )

    parser.add_argument(
        "-hp",
        "--hyper_params",
        default="test01/quick_4gpu_resnet50_simclr.yaml",
        help="Path to the hyper parameters config file.\n",
    )

    return parser
