'''
Project: SelfSupSurg
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''
from typing import Any, Dict

import numpy as np
import random
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from PIL import Image


@register_transform("ImgPilRandomResizeLargerSide")
class ImgPilRandomResizeLargerSide(ClassyTransform):
    def __init__(
        self,
        lower_bound: int,
        upper_bound: int,
    ):
        """
        Randomly Resizes the larger side between "lower_bound" and "upper_bound". Note that the torchvision.Resize transform
        crops the smaller edge to the provided size and is unable to crop the larger
        side.
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, img):
        # Resize the longest side to self.size.
        img_size_hw = np.array((img.size[1], img.size[0]))
        size = random.randint(self.lower_bound, self.upper_bound)
        ratio = float(size) / np.max(img_size_hw)
        new_size = tuple(np.round(img_size_hw * ratio).astype(np.int32))
        img_resized = img.resize((new_size[1], new_size[0]), Image.BILINEAR)

        return img_resized

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgPilResizeLargerSide":
        """
        Instantiates ImgPilRandomSolarize from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            ImgPilRandomSolarize instance.
        """
        lower_bound = config.get("lower_bound", 1024)
        upper_bound = config.get("upper_bound", 1024)

        return cls(lower_bound, upper_bound)
