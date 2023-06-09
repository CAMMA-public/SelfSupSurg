'''
Project: SelfSupSurg
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''

import os
import json
import time
import copy
import numpy as np

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets

class FCN(torch.nn.Module):
    def __init__(self, hparams):
        super(FCN, self).__init__()
        self.linear = torch.nn.Linear(
                        hparams.FCN.in_features,
                        hparams.FCN.out_features
                    )

    def forward(self, x, mask):
        outputs = self.linear(x)
        return outputs
