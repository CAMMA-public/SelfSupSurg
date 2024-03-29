'''
Project: SelfSupSurg
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''

from typing import List, Union

import torch
from classy_vision.generic.util import is_on_gpu
from classy_vision.losses import ClassyLoss, register_loss
from torch import nn
from vissl.config import AttrDict


@register_loss("weighted_bce_logits_multiple_output_multiple_target")
class WeightedBCELogitsMultipleOutputMultipleTargetLoss(ClassyLoss):
    def __init__(self, loss_config: AttrDict):
        """
        Intializer for the sum cross-entropy loss. For a single
        tensor, this is equivalent to the cross-entropy loss. For a
        list of tensors, this computes the sum of the cross-entropy
        losses for each tensor in the list against the target.

        Config params:
            reduction: specifies reduction to apply to the output, optional
            normalize_output: Whether to L2 normalize the outputs
            world_size: total number of gpus in training. automatically inferred by vissl
        """
        super(WeightedBCELogitsMultipleOutputMultipleTargetLoss, self).__init__()
        self.loss_config = loss_config
        self._reduction = loss_config.get("reduction", "mean")
        self._normalize_output = loss_config.get("normalize_output", False)
        # self._world_size = loss_config["world_size"]
        self._pos_weight = torch.tensor(loss_config["pos_weight"]).cuda()      
        self._losses = nn.BCEWithLogitsLoss(reduction=self._reduction, pos_weight=self._pos_weight)  
        # self._losses = nn.MultiLabelSoftMarginLoss(reduction=self._reduction, weight=self._pos_weight)
        self._losses.cuda()

    @classmethod
    def from_config(cls, loss_config: AttrDict):
        """
        Instantiates WeightedBCELogitsMultipleOutputMultipleTargetLoss from configuration.

        Args:
            loss_config: configuration for the loss

        Returns:
            WeightedBCELogitsMultipleOutputMultipleTargetLoss instance.
        """
        return cls(loss_config)

    # def _create_loss_function(self):
    #     copy_to_gpu = is_on_gpu(self._losses)
    #     self._losses.append(nn.MultiLabelSoftMarginLoss(reduction=self._reduction, weight=self._pos_weight))
    #     if copy_to_gpu:
    #         self._losses.cuda()
    #     return self

    def forward(
        self, output: torch.Tensor, target: torch.Tensor
    ):
        """
        For each output and single target, loss is calculated.
        The returned loss value is the sum loss across all outputs.
        """
        loss = self._losses(output, target.float())
        # if isinstance(output, torch.Tensor):
        #     output = [output]
        # assert isinstance(
        #     output, list
        # ), "Model output should be a list of tensors. Got Type {}".format(type(output))
        # assert torch.is_tensor(target), "Target should be a tensor. Got Type {}".format(
        #     type(target)
        # )

        # loss = 0
        # # for idx, pred in enumerate(output):
        # outputs = output
        # for idx, pred in enumerate(outputs):
        #     normalized_pred = pred
        #     if self._normalize_output:
        #         normalized_pred = nn.functional.normalize(pred, dim=1, p=2)

        #     mask1 = target == -1
        #     # number of valid (0 or 1 label) entries per class
        #     num_per_class = torch.sum(~mask1, dim=0)
        #     # number of classes with no valid entries.
        #     mask2 = num_per_class == 0
        #     num_per_class.masked_fill_(mask2, 1)

        #     if idx >= len(self._losses):
        #         self._create_loss_function()

            
            # loss += torch.sum(
            #     self._losses[idx](normalized_pred, target.float()).masked_fill_(
            #         mask1, 0
            #     )
            #     / num_per_class
            # )
        return loss
