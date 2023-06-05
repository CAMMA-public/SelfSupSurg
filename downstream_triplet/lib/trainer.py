"""
Trainer class to support model training and inference on CholecT50 dataset
"""

import os
import time
import random
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim

from pprint import pprint, pformat
from .evaluator import evaluate
from .dataloader import CholecT50
from .utils import *
from .build_optims import init_optims
from .models import get_model

class TripletTrainer:
    def __init__(self, args):
        self.args = args
        self.best_metric = 0.0
        self.patience = 0
        self.get_records()
        self.aug_list = self.args.TRAIN.AUG_LIST

        # load model from the weights either as ssl pretrained weights or plain imagenet
        self.model = get_model(args)

        # init optimizers and schedulers
        self.optimizer, self.scheduler = init_optims(self.args, self.model)

        # put the args to config file
        logging.info('>>>>>>>>>>>>>>>>>>>>>>  Args Start <<<<<<<<<<<<<<<<<<<<<<<\n') 
        logging.info(pformat(vars(args)))        

    def get_records(self):
        self.train_records, self.val_records, self.test_records = get_video_list(self.args)
        logging.info("======== Train videos ========")
        logging.info(self.train_records)
        logging.info("======== Val videos ==========")
        logging.info(self.val_records)
        logging.info("======== Test videos =========")
        logging.info(self.test_records)
        logging.info("==============================")

    def train(self):
        self.model.train()
        self.model.module.backbone.eval()

        logging.info("Training started..")

        # make a copy of the aug list
        aug_list_loop = self.aug_list.copy()

        for epoch in range(self.args.TRAIN.MAX_EPOCHS):
            # get loaders and augs
            train_aug = aug_list_loop.pop()
            if len(aug_list_loop) == 0:
                aug_list_loop = self.aug_list.copy()
                np.random.shuffle(aug_list_loop)

            self.train_loader = CholecT50(self.args.DATA.HOME, self.train_records,
                                        split="train", 
                                        aug=train_aug)(batch_size=self.args.TRAIN.BS, num_workers=self.args.TRAIN.NW, shuffle=True)

            train_loss = self.train_one_epoch(epoch=epoch)

            # start checkpoint
            flag = self.save_checkpoints(epoch)

            if flag:
                logging.info("Training stopped!")
                break

        logging.info("Training is complete!")

        return train_loss

    def train_one_epoch(self, epoch):
        loss_tracker =  AverageMeter()
        tqdm_loader = tqdm(self.train_loader, unit="batch")
        for i, (frames, (y_i, y_v, y_t, y_ivt)) in enumerate(tqdm_loader):
            y_i    = y_i.float().cuda()
            y_v    = y_v.float().cuda()
            y_t    = y_t.float().cuda()
            y_ivt  = y_ivt.float().cuda()
            frames = frames.cuda()

            ivt_logit = self.model(frames)
            loss = bce_loss(ivt_logit, y_ivt, None)
            
            self.optimizer.zero_grad(set_to_none=True)
            loss_tracker.update(loss.item())
            
            loss.backward()
            self.optimizer.step()
            self.model.zero_grad()

            tqdm_loader.set_postfix(mode='TRAIN', epoch=epoch, batch=i, loss=f"{loss_tracker.avg:.3f}")
        
        self.scheduler.step()

        return loss

    # save checkpoints if val improved ..
    def save_checkpoints(self, epoch):
        # evaluate on val
        logging.info(f"Starting evaluation on val data ...")
        val_results = evaluate(self.model, self.args, self.val_records, mode="eval")
        val_ivtmAP  = val_results['triplet_mAP']

        # save ckpt if val metric improves ..
        if val_ivtmAP > self.best_metric:
            self.best_metric = val_ivtmAP
            logging.info(f"Starting evaluation on test data ...")
            test_results = evaluate(self.model, self.args, self.test_records, mode="eval")
            test_ivtmAP  = test_results['triplet_mAP']

            # save the weights
            ckp_save_name = os.path.join(
                                    self.args.MODEL.SAVEDIR,
                                    self.args.en,
                                    f'{self.args.en}_{epoch}.pth'
                                )
            model_state = {
                         'model': self.model.state_dict(),
                    }

            torch.save(model_state, ckp_save_name)
            self.patience = 0
        else:
            self.patience += 1

        if self.patience > self.args.TRAIN.ES_PATIENCE:
            logging.info("Val metric plateaued .. stopping training!!")
            self.patience = 0 
            return True
        else:
            return False

    def test(self):
        # load checkpoints and evaluate
        checkpoint_name = os.path.join(self.args.MODEL.SAVEDIR, f'{self.args.ckp_n}_final.pth')
        logging.info(f"Loading checkpoint from >>> {checkpoint_name}")

        assert os.path.exists(checkpoint_name)

        load_ckpt(self.model, checkpoint_name)
        logging.info("Checkpoint is successfully loaded!")

        # evaluate on test records
        test_results = evaluate(self.model, self.args, self.test_records, mode="eval")


if __name__ == "__main__":
    pass

