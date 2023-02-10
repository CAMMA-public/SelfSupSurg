'''
Project: SelfSupSurg
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''

import os
import h5py
import random
import pickle
import numpy as np

from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import logging
logging.basicConfig(level=logging.INFO)

class FeatureDataset(Dataset):
    '''Data loader for Linear/TCN eval'''

    def __init__(self, cfg=None, data_split='train', transform=None):
        self.cfg = cfg
        self.ids = []
        self.videos = []
        self.images = []
        self.labels = []
        self.data_split = data_split
        self.model_name = self.cfg.MODEL.name
        self.task = self.cfg.DATA.LABEL_TOOL_OR_PHASE
        self.n_phases = self.cfg.DATA.NUM_CLASSES_PHASE
        self.n_tools = self.cfg.DATA.NUM_OF_SURGICAL_TOOLS
        self.data_loaded = False
        self.load_data()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        batch_label = self.labels[idx]
        batch_ids = self.ids[idx]

        if self.model_name == 'FCN':
            batch_input = self.images[idx]
            return batch_ids, batch_input, batch_label, []

        batch_input = [self.videos[idx]]
        if len(batch_label.size()) == 1:
            batch_label = batch_label.unsqueeze(0)
        if len(batch_ids.size()) == 1:
            batch_ids = batch_ids.unsqueeze(0)

        length_of_sequences = map(len, batch_label)

        max_seq = np.shape(batch_input[0])[0] #max(length_of_sequences)
        batch_input_tensor = torch.zeros(len(batch_input), batch_input[0].size()[1], batch_input[0].size()[0], dtype=torch.float).to('cuda')
        batch_label_tensor = torch.ones(len(batch_input), max_seq, dtype=torch.long).to('cuda')*(-1)
        mask = torch.zeros(len(batch_input), self.n_phases, max_seq, dtype=torch.float).to('cuda')
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :batch_input[i].size()[0]] = torch.transpose(batch_input[i],0,1)
            batch_label_tensor[i, :batch_input[i].size()[0]] = batch_label[i]
            mask[i, :, :np.shape(batch_input[i])[0]] = torch.ones(self.n_phases, batch_input[i].size()[0])

        return torch.squeeze(batch_ids,0), torch.squeeze(batch_input_tensor,0), torch.squeeze(batch_label_tensor,0), torch.squeeze(mask,0)

    def load_data(self):
        data_path = self.cfg.DATA.TRAIN.DATA_PATHS
        label_key = self.task + '_gt'
        data_split = self.data_split #'val' if self.data_split == 'test' else self.data_split
        feat_path = self.cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE
        if os.path.isfile(feat_path): feat_path = os.path.dirname(feat_path)
        if os.path.exists(os.path.join(feat_path,'extracted_features_Trunk')):
            feat_folder = 'extracted_features_Trunk'
        else:
            feat_folder = 'extracted_features_Head'

        logging.info('feat_path: '+str(feat_path))
        logging.info('extracted_features_Trunk: '+str(os.path.exists(os.path.join(feat_path,'extracted_features_Trunk'))))
        logging.info('extracted_features_Head: '+str(os.path.exists(os.path.join(feat_path,'extracted_features_Head'))))
        feat_file = os.path.join(
                        feat_path,
                        feat_folder,
                        'extracted_features_{}.hdf5'.format(data_split.lower())
                    )
        if self.data_split == 'val':
            data_path = self.cfg.DATA.VAL.DATA_PATHS
        elif self.data_split == 'test':
            data_path = self.cfg.DATA.TEST.DATA_PATHS

        if os.path.exists(feat_file):
            feature_data = h5py.File(feat_file, 'r')
        else:
            self.data_loaded = False
            return
        with open(data_path[0], "rb") as fp:
            data = pickle.load(fp)
        vids = {vname: data[vname][0]['unique_id'] // 10 ** 8 for vname in data.keys()}
        videos, images, labels, ids = [], [], [], []
        for vname, vid in vids.items():
            inds = np.argwhere(feature_data['video_id'][:] == vid).flatten()
            _, order = np.unique(feature_data['frame_id'][inds], return_index=True)
            inds = inds[order]
            assert len(inds) != 0, "video missing in extracted features "+str(vid)
            #print('feature_data', feature_data['frame_id'][:][inds[:10]])
            #print(np.array([item['Frame_id'] for item in data[vname][:10]]))
            #assert np.all(feature_data['frame_id'][:][inds[:10]] == np.array([item['Frame_id'] for item in data[vname][:10]])), "indexs doesn't match: pickle vs hdf5"
            videos.append(feature_data['embeddings'][:][inds])
            ids.append(feature_data['frame_id'][:][inds] + vid * 10 ** 8)
            labels.append(np.array([
                    np.array(item[label_key]).astype(np.int64)
                    if item[label_key] is not None
                    else np.array([0] * self.n_tools)
                    for item in data[vname]
            ]))
            assert len(videos[-1]) == len(labels[-1]), "Dimension mismatch: len(video) != len(label): "+str(len(videos[-1]))+" "+str(len(labels[-1]))

        tensorType = torch.FloatTensor if self.task == 'Tool' else torch.LongTensor
        if self.model_name == 'FCN':
            self.images = torch.from_numpy(np.concatenate(videos)).to('cuda')
            self.labels = tensorType(np.concatenate(labels)).to('cuda')
            self.ids = torch.from_numpy(np.concatenate(ids)).to('cuda')
        if self.model_name == 'TCN':
            self.videos = [torch.from_numpy(v).to('cuda') for v in videos]
            self.labels = [torch.from_numpy(np.stack(l)).to('cuda') for l in labels]
            self.ids = [torch.from_numpy(i).to('cuda') for i in ids]

        self.data_loaded = True
        return


def create_data_loaders(cfg, shuffle=False):
    n_workers = 0
    minibatch = cfg.DATA.TRAIN.BATCHSIZE_PER_REPLICA
    if cfg.MODEL.name == 'TCN': minibatch = 1
    datasets = FeatureDataset(cfg, data_split='train')
    logging.info("train data loaded: " + str(datasets.data_loaded))
    if datasets.data_loaded:
        train_loader = DataLoader(
                            datasets, batch_size=minibatch,
                            shuffle=shuffle, num_workers=n_workers, drop_last=False
                        )
    else:
        train_loader = None

    datasets = FeatureDataset(cfg, data_split='test')
    logging.info("test data loaded: " + str(datasets.data_loaded))
    if datasets.data_loaded:
        test_loader = DataLoader(
                        datasets, batch_size=minibatch,
                        shuffle=shuffle, num_workers=n_workers, drop_last=False
                    )
    else:
        test_loader = None


    datasets = FeatureDataset(cfg, data_split='val')
    logging.info("val data loaded: " + str(datasets.data_loaded))
    if datasets.data_loaded:
        val_loader = DataLoader(
                            datasets, batch_size=minibatch,
                            shuffle=shuffle, num_workers=n_workers, drop_last=False
                        )
    else:
        val_loader = None

    return train_loader, test_loader, val_loader

