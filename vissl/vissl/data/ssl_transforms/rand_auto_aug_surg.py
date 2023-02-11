'''
Project: SelfSupSurg
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''

import numpy as np
from classy_vision.dataset.transforms import register_transform
from vissl.data.ssl_transforms.rand_auto_aug import RandAugment
from vissl.data.ssl_transforms.rand_auto_aug import (
    rand_augment_ops,
    _select_rand_weights,
)

_FILL = (128, 128, 128)
_HPARAMS_DEFAULT = {"translate_const": 250, "img_mean": _FILL}
@register_transform("RandAugmentSurgery")
class RandAugmentSurgery(RandAugment):
    """
    Create a RandAugmentSurgery transform.
    :param magnitude: integer magnitude of rand augment
    :param magnitude_std: standard deviation of magnitude. If > 0, introduces
    random variability in the augmentation magnitude.
    :param num_layers: integer number of transforms
    :param weight_choice: Index of pre-determined probability distribution
    over augmentations. Currently only one such distribution available (i.e.
    no valid values other than 0 or None), unclear if beneficial. Default =
    None.
    """
    def __init__(
        self,
        magnitude=10,
        magnitude_std=0,
        num_layers=2,
        weight_choice=None,
        transforms=[],
        **kwargs
    ):
        hparams = kwargs
        hparams.update(_HPARAMS_DEFAULT)
        hparams["magnitude_std"] = magnitude_std
        self.num_layers = num_layers
        self.choice_weights = (
            None if weight_choice is None else _select_rand_weights(weight_choice)
        )
        self.ops = rand_augment_ops(
            magnitude=magnitude, hparams=hparams, transforms=transforms
        )
