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
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from downstream_phase_tcn.linear_evaluation import FCN
from downstream_phase_tcn.tcn import MultiStageModel

from utils.model_helpers import save_results
from utils.model_helpers import init_loss, init_optimizer, init_metric

import logging
logging.basicConfig(level=logging.INFO)

class LinearEvalTrainer:
    def __init__(self, cfg):
        self.stats = {}
        self.cfg = cfg
        self.model = FCN(self.cfg.MODEL)
        self.loss = init_loss(self.cfg.LOSS)
        self.optimizer, self.scheduler = init_optimizer(self.cfg.OPTIMIZER, self.model)
        self.metric = init_metric(self.cfg.METERS)

    def add_data_loaders(self, train_ds, test_ds, val_ds):
        self.train_loader = train_ds
        self.test_loader = test_ds
        self.val_loader = val_ds

    def loss_func(self, labels, predicts):
        return self.loss(labels, predicts)

    def train_epoch(self):
        train_loss = 0.0
        train_metric = 0.0

        for bidx, batch in enumerate(self.train_loader):
            _, batch_input, batch_label, mask = batch
            outputs = self.model(batch_input, mask)
            loss = self.loss_func(outputs, batch_label)
            train_loss += loss.item()
            train_metric += self.metric(batch_label, outputs)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()
        num_train_batch = len(self.train_loader)
        train_loss /= num_train_batch
        train_metric = 100 * train_metric / num_train_batch

        return train_loss, train_metric

    def test_epoch(self):
        test_loss = 0.0
        test_metric = 0.0
        val_loss = 0.0
        val_metric = 0.0

        if self.test_loader:
            for bidx, batch in enumerate(self.test_loader):
                _, batch_input, batch_label, mask = batch
                outputs = self.model(batch_input, mask)
                loss = self.loss_func(outputs, batch_label)
                test_loss += loss.item()
                test_metric += self.metric(batch_label, outputs)

            num_test_batch = len(self.test_loader)
            test_loss /= num_test_batch
            test_metric = 100 * test_metric / num_test_batch

        if self.val_loader:
            for bidx, batch in enumerate(self.val_loader):
                _, batch_input, batch_label, mask = batch
                outputs = self.model(batch_input, mask)
                loss = self.loss_func(outputs, batch_label)
                val_loss += loss.item()
                val_metric += self.metric(batch_label, outputs)

            num_val_batch = len(self.val_loader)
            val_loss /= num_val_batch
            val_metric = 100 * val_metric / num_val_batch


        return val_loss, val_metric, test_loss, test_metric

    def train(self, device='cuda'):
        logging.info('Starting Training!!!')
        stats = []
        self.model.train()
        self.model.to(device)
        self.loss.to(device)
        ckpt_dir = self.cfg.CHECKPOINT.DIR
        save_freq = self.cfg.CHECKPOINT.CHECKPOINT_FREQUENCY
        metric_name = self.cfg.METERS.name
        num_epochs = self.cfg.OPTIMIZER.num_epochs
        for epoch in range(num_epochs):
            train_loss, train_metric = self.train_epoch()
            val_loss, val_metric, test_loss, test_metric = self.test_epoch()

            if epoch % save_freq == 0 or epoch == num_epochs-1:
                torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "epoch-" + str(epoch) + ".model"))
                torch.save(self.optimizer.state_dict(), os.path.join(ckpt_dir, "epoch-" + str(epoch) + ".opt"))

            logging.info("[epoch %3d]: train loss = %0.6f, val loss = %0.6f, test loss = %0.6f, train %s = %2.3f, val %s = %2.3f, test %s = %2.3f" %(
                epoch, train_loss, val_loss, test_loss, metric_name, train_metric, metric_name, val_metric, metric_name, test_metric
            ))
            stats.append({
                "iteration": epoch, "phase_idx": epoch*2,
                "train_"+metric_name: {"top_1": {"0": train_metric}},
                "train_phase_idx": epoch
            })
            stats.append({
                "iteration": epoch, "phase_idx": epoch*2+1,
                "test_"+metric_name: {"top_1": {"0": test_metric}},
                "train_phase_idx": epoch
            })
            stats.append({
                "iteration": epoch, "phase_idx": epoch*2+2,
                "val_"+metric_name: {"top_1": {"0": val_metric}},
                "train_phase_idx": epoch
            })
            ## Save training stats after each epoch
            with open(os.path.join(ckpt_dir, 'metrics.json'), 'w') as fp:
                json.dump(stats, fp)

        logging.info('Finishing Training!!!')
        return True

    def test(self, data_splits=['test', 'val'], device='cuda'):
        run_dir = self.cfg.RUN_DIR
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            dsets = ['train', 'val', 'test']
            datasets = [self.train_loader, self.val_loader, self.test_loader]
            for i, dsplit in enumerate(dsets):
                if dsplit not in data_splits:
                    continue
                logging.info('-' * 80)
                if datasets[i]:
                    logging.info(('Extracting features: '+dsplit).center(80))
                    inds, predictions, targets = [], [], []
                    for bidx, batch in enumerate(datasets[i]):
                        batch_inds, batch_input, batch_label, mask = batch
                        outputs = self.model(batch_input, mask)
                        inds.append(batch_inds.cpu().data.numpy())
                        predictions.append(outputs.cpu().data.numpy())
                        targets.append(batch_label.cpu().data.numpy())

                    inds = np.concatenate(inds)
                    predictions =np.concatenate(predictions)
                    targets =np.concatenate(targets)
                    save_results(run_dir, inds, predictions, targets)
                    logging.info(('Completed: Extracting features: '+dsplit).center(80))
                    logging.info('-' * 80)

        return True

class TeCNOTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        tcn_cfg = self.cfg.MODEL.TCN
        self.model = MultiStageModel(
            tcn_cfg.MSTCN_STAGES, tcn_cfg.MSTCN_LAYERS, tcn_cfg.MSTCN_F_MAPS,
            tcn_cfg.MSTCN_F_DIM, tcn_cfg.OUT_FEATURES
        )

        self.n_classes = tcn_cfg.OUT_FEATURES
        self.loss = init_loss(cfg.LOSS)
        self.mse = nn.MSELoss(reduction='none')
        self.optimizer, self.scheduler = init_optimizer(self.cfg.OPTIMIZER, self.model)
        self.metric = init_metric(self.cfg.METERS)
        self.best_model_weights = None

    def add_data_loaders(self, train_ds, test_ds, val_ds):
        self.train_loader = train_ds
        self.test_loader = test_ds
        self.val_loader = val_ds

    def loss_func(self, labels, predicts, mask):
        # loss = self.loss(labels, predicts)
        loss = 0.0
        for p in predicts:
            loss += self.loss(p.transpose(2, 1).contiguous().view(-1, self.n_classes), labels.view(-1))
            loss += 0.15 * torch.mean(torch.clamp(self.mse(
                        F.log_softmax(p[:, :, 1:], dim=1),
                        F.log_softmax(p.detach()[:, :, :-1], dim=1)
                    ), min=0, max=16)*mask[:, :, 1:])

        return loss

    def train_epoch(self):
        train_loss = 0.0
        train_metric = 0.0
        num_train_total = 0
        if self.train_loader:
            for bidx, batch in enumerate(self.train_loader):
                _, batch_input, batch_label, mask = batch
                outputs = self.model(batch_input, mask)
                loss = self.loss_func(batch_label, outputs, mask)
                train_loss += loss.item()
                batch_numel = batch_label.squeeze().shape[0]
                train_metric += (batch_label.squeeze() == outputs[-1].squeeze().argmax(0)).float().sum().detach().cpu().item()
                num_train_total += batch_numel

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # torch.cuda.synchronize()

            if self.scheduler is not None:
                self.scheduler.step()

            train_loss /= len(self.train_loader)
            train_metric /= num_train_total

        return train_loss, train_metric

    def val_epoch(self):
        self.model.eval()
        val_loss = 0.0
        val_metric = 0.0
        num_val_total = 0
        for bidx, batch in enumerate(self.val_loader):
            _, batch_input, batch_label, mask = batch
            # batch_input, batch_label = batch_input.cuda(), batch_label.cuda()
            outputs = self.model(batch_input, mask)
            loss = self.loss_func(batch_label, outputs, mask)
            val_loss += loss.item()
            batch_numel = batch_label.squeeze().shape[0]
            val_metric += (batch_label.squeeze() == outputs[-1].squeeze().argmax(0)).float().sum().detach().cpu().item()
            num_val_total += batch_numel

        val_loss /= len(self.val_loader)
        val_metric /= num_val_total

        return val_loss, val_metric

    def test_epoch(self):
        self.model.eval()
        test_loss = 0.0
        test_metric = 0.0
        num_test_total = 0
        for bidx, batch in enumerate(self.test_loader):
            _, batch_input, batch_label, mask = batch
            # batch_input, batch_label = batch_input.cuda(), batch_label.cuda()
            outputs = self.model(batch_input, mask)
            loss = self.loss_func(batch_label, outputs, mask)
            test_loss += loss.item()
            batch_numel = batch_label.squeeze().shape[0]
            test_metric += (batch_label.squeeze() == outputs[-1].squeeze().argmax(0)).float().sum().detach().cpu().item()
            num_test_total += batch_numel

        test_loss /= len(self.test_loader)
        test_metric /= num_test_total

        return test_loss, test_metric

    def train(self, device='cuda'):
        '''
        Model training
        '''
        logging.info('Starting Training!!!')
        stats = []
        self.model.train()
        self.model.to(device)
        self.loss.to(device)
        self.mse.to(device)
        ckpt_dir = self.cfg.CHECKPOINT.DIR
        save_freq = self.cfg.CHECKPOINT.CHECKPOINT_FREQUENCY
        metric_name = self.cfg.METERS.name
        num_epochs = self.cfg.OPTIMIZER.num_epochs
        best_epoch = 0
        max_val_metric = 0

        for epoch in range(num_epochs):
            train_loss, train_metric = self.train_epoch()
            val_loss, val_metric = self.val_epoch()

            if epoch % save_freq == 0 or epoch == num_epochs-1:
                torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "epoch-" + str(epoch) + ".model"))
                torch.save(self.optimizer.state_dict(), os.path.join(ckpt_dir, "epoch-" + str(epoch) + ".opt"))

            if val_metric > max_val_metric:
                best_epoch = epoch
                max_val_metric = val_metric
                self.best_model_weights = copy.deepcopy(self.model.state_dict())
                torch.save(self.best_model_weights, os.path.join(ckpt_dir, "epoch-" + str(epoch) + ".model"))
                torch.save(self.optimizer.state_dict(), os.path.join(ckpt_dir, "epoch-" + str(epoch) + ".opt"))

            logging.info("[epoch %3d]: train loss = %0.6f, test loss = %0.6f, train %s = %2.3f, test %s = %2.3f" %(
                epoch, train_loss, val_loss, metric_name, train_metric, metric_name, val_metric
            ))
            stats.append({
                "iteration": epoch, "phase_idx": epoch*2,
                "train_"+metric_name: {"top_1": {"0": train_metric}},
                "train_phase_idx": epoch
            })
            stats.append({
                "iteration": epoch, "phase_idx": epoch*2+1,
                "val_"+metric_name: {"top_1": {"0": val_metric}},
                "train_phase_idx": epoch
            })
            ## Save training stats after each epoch
            with open(os.path.join(ckpt_dir, 'metrics.json'), 'w') as fp:
                json.dump(stats, fp)

        logging.info('Best Epoch: {}, Best Val Metric Value: {}'.format(best_epoch, max_val_metric))
        logging.info('Finishing Training!!!')
        return True

    def test(self, data_splits=['test'], device='cuda'):
        run_dir = self.cfg.RUN_DIR

        # load best checkpoint
        self.model.load_state_dict(self.best_model_weights)
        self.model.eval()

        with torch.no_grad():
            self.model.to(device)
            dsets = ['train', 'val', 'test']
            datasets = [self.train_loader, self.val_loader, self.test_loader]
            for i, dsplit in enumerate(dsets):
                if dsplit not in data_splits:
                    continue
                logging.info('-' * 80)
                logging.info(('Extracting features: '+dsplit).center(80))
                inds, predictions, targets = [], [], []
                for bidx, batch in enumerate(datasets[i]):
                    batch_inds, batch_input, batch_label, mask = batch
                    outputs = self.model(batch_input, mask)
                    inds.append(batch_inds.cpu().data.numpy())
                    predictions.append(outputs.cpu().data.numpy())
                    targets.append(batch_label.cpu().data.numpy())

                # concatenate lists and save (save output of last stage)
                inds = np.concatenate(inds, axis=-1).squeeze()
                predictions = np.concatenate(predictions, axis=-1).squeeze()[-1].T
                targets = np.concatenate(targets, axis=-1).squeeze()
                save_results(run_dir, inds, predictions, targets)
                logging.info(('Completed: Extracting features: '+dsplit).center(80))
                logging.info('-' * 80)
                acc = (predictions.argmax(1) == targets).mean()
                logging.info('{} Accuracy: {}'.format(dsplit, acc))
                logging.info('-' * 80)

        return True
