'''
Project: SelfSupSurg
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''

import os
import torch
import numpy as np
import torch.nn as nn
from torch import optim

from sklearn.metrics import average_precision_score
np.seterr(divide='ignore', invalid='ignore')

def init_optimizer(hparams, model):
    optimizer, scheduler = None, None
    if hparams.name == 'adam':
        optimizer = optim.Adam(
                        model.parameters(),
                        lr=hparams.param_schedulers.lr.base_lr,
                    )
        scheduler = None

    elif hparams.name == 'adamw':
        optimizer = optim.AdamW(
                        model.parameters(),
                        lr=hparams.param_schedulers.lr.base_lr,
                    )
        scheduler = None

    elif hparams.name == 'sgd':
        optimizer = optim.SGD(
                        model.parameters(),
                        lr=hparams.param_schedulers.lr.base_lr,
                        momentum=hparams.momentum,
                        weight_decay=hparams.weight_decay,
                        nesterov=hparams.nesterov
                    )

        scheduler = optim.lr_scheduler.MultiStepLR(
                        optimizer,
                        milestones=hparams.param_schedulers.lr.milestones,
                        gamma=hparams.param_schedulers.lr.gamma,
                    )

    return optimizer, scheduler

def init_loss(hparams):
    loss_func = None
    if hparams.name == 'cross_entropy_multiple_output_single_target':
        loss_params = hparams.cross_entropy_multiple_output_single_target
        loss_func = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(loss_params.pos_weight),
            ignore_index=loss_params.ignore_index
        )
    if hparams.name == 'bce_logits_multiple_output_single_target':
        loss_params = hparams.bce_logits_multiple_output_single_target
        loss_func = nn.BCEWithLogitsLoss(
            pos_weight=torch.FloatTensor(loss_params.pos_weight),
        )
    return loss_func

def init_metric(hparams):
    metric = None
    if hparams.name == 'accuracy_list_meter':
        metric = accuracy
    if hparams.name == 'mAP':
        metric = mAP
    return metric

def accuracy(labels, predictions, mask=None):
    t, predicted = torch.max(predictions, 1)
    acc = (predicted == labels).float()
    if mask != None:
        acc =  (acc * mask[:, 0, :].squeeze(1)).sum().item()
    else:
        acc = acc.mean().item()
    return acc

def tensor_to_np_mAP(targets, predicts):
    targets = labels.cpu().numpy()
    predicts = torch.sigmoid(predictions).cpu().numpy()
    return targets, predicts

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mAP(labels, predictions, mean=True, istensor=True):
    if istensor:
        labels = labels.detach().cpu().numpy()
        predictions = predictions.detach().cpu().numpy()
    metrics = np.array(average_precision_score(labels, predictions, average=None))
    if mean: metrics = np.sum([x for x in metrics if x==x])/len(metrics)
    return metrics

def save_results(path, inds, preds, targets):
    path = os.path.join(path, 'extracted_features')
    nps = [inds, preds, targets]
    fnames = ['inds', 'features', 'targets']
    if not os.path.exists(path): os.makedirs(path)
    for i, arr in enumerate(nps):
        print('Saving predictions:', fnames[i], arr.shape)
        with open(os.path.join(path, '{}_all.npy'.format(fnames[i])), 'wb') as f:
            np.save(f, arr)
    return

class mAP_Accumulator(object):
    def __init__(self, num_class=6):
        self.num_class = num_class
        self.reset()

    def update(self, tragets, predictions):
        self.predictions = np.append(self.predictions, predictions, axis=0)
        self.tragets = np.append(self.tragets, tragets, axis=0)

    def reset(self):
        self.predictions = np.empty(shape = [0,self.num_class], dtype=np.float)
        self.tragets = np.empty(shape = [0,self.num_class], dtype=np.int)

    def compute(self):
        computed_ap = average_precision_score(self.tragets, self.predictions, average=None)
        actual_ap   = np.mean([x for x in computed_ap if x==x])
        return actual_ap
