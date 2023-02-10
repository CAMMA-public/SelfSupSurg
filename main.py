'''
Project: SelfSupSurg
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''


"""
Wrapper to call torch.distributed.launch to run multi-gpu trainings.
Supports two engines: train and extract_features.
Supports SLURM as an option. Set config.SLURM.USE_SLURM=true to use slurm.
"""
import os
if "SCRATCH" in os.environ:
    os.environ["TMPDIR"] = os.path.join(os.environ["SCRATCH"], 'tmp')

import gc
import sys
from typing import Any, List
import torch

import utils.config_overrides as cov
from hydra.experimental import compose, initialize_config_module
from vissl.utils.distributed_launcher import (
    launch_distributed,
    launch_distributed_on_slurm,
)
from vissl.utils.hydra_config import convert_to_attrdict, is_hydra_available
from vissl.utils.slurm import is_submitit_available
from vissl.data.dataset_catalog import VisslDatasetCatalog


def hydra_main(overrides: List[Any], mode="self_supervised"):
    ######################################################################################
    # DO NOT MOVE THIS IMPORT TO TOP LEVEL: submitit processes will not be initialized
    # correctly (MKL_THREADING_LAYER will be set to INTEL instead of GNU)
    ######################################################################################
    from vissl.hooks import default_hook_generator

    ######################################################################################

    print(f"####### overrides: {overrides}")
    with initialize_config_module(config_module="vissl.config"):
        cfg = compose("defaults", overrides=overrides)

    args, config = convert_to_attrdict(cfg)

    if mode == "feature_extraction":
        config["DATA"]["TRAIN"]["TRANSFORMS"] = cfg["config"]["DATA"]["TEST"]["TRANSFORMS"]
        config["DATA"]["TRAIN"]["COLLATE_FUNCTION"] = "default_collate"
        config["DATA"]["TEST"]["COLLATE_FUNCTION"] = "default_collate"

    if config.SLURM.USE_SLURM:
        assert (
            is_submitit_available()
        ), "Please 'pip install submitit' to schedule jobs on SLURM"
        launch_distributed_on_slurm(engine_name=args.engine_name, cfg=config)
    else:
        launch_distributed(
            cfg=config,
            node_id=args.node_id,
            engine_name=args.engine_name,
            hook_generator=default_hook_generator,
        )


if __name__ == "__main__":
    """
    Example usage:

    `python tools/run_distributed_engines.py config=test/integration_test/quick_simclr`
    """
    
    #VisslDatasetCatalog.register_data(name="surgery_datasets", data_dict={"train": "dummy", "val":"dummy", "test":"dummy"})
    # torch.set_deterministic(True)
    parser = cov.create_argument_parser()
    args = parser.parse_args()

    config_file = args.hyper_params
    training_mode = args.mode
    split = args.split
    feature = args.feature
    print('Mode:', training_mode)

    override_fn = {
        'supervised': cov.supervised_overrides,
        'self_supervised': cov.ssl_overrides,
        'feature_extraction': cov.extract_features_overrides,
    }

    print('=' * 80)
    print(('Training Mode: ' + training_mode).center(80))
    print('=' * 80)
    overrides = override_fn[training_mode](config_file, split, feature=feature)

    assert is_hydra_available(), "Make sure to install hydra"
    hydra_main(overrides=overrides, mode=training_mode)
