'''
Project: SelfSupSurg
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''


import os
from shutil import copy2
import glob
import torch
import random
import numpy as np
import h5py
from utils.file_helpers import parse_config, create_argument_parser
from downstream_phase_tcn.trainers import LinearEvalTrainer, TeCNOTrainer
from downstream_phase_tcn.data_loader import create_data_loaders

import logging
logging.basicConfig(level=logging.INFO)

def collect_embeddings(path):
    splits = ['train', 'test', 'val']
    for split in splits:
        feat_path = os.path.join(path, split)
        if not os.path.exists(feat_path): continue
        if os.path.exists(os.path.join(
            path, 'extracted_features_{}.hdf5'.format(split))):
            print('embeddings already collected:', feat_path)
            continue
        print('collecting embeddings:', feat_path)
        finds = sorted(glob.glob(os.path.join(
                feat_path, '*_inds.npy'
            )))
        ftargets = sorted(glob.glob(os.path.join(
                    feat_path, '*_targets.npy'
                )))
        ffeatures = sorted(glob.glob(os.path.join(
                    feat_path, '*_features.npy'
                )))
        inds = np.concatenate([np.load(f) for f in finds])
        targets = np.concatenate([np.load(f) for f in ftargets])
        features = np.concatenate([np.load(f) for f in ffeatures])

        order = np.argsort(inds)
        inds = inds[order]
        targets = targets[order]
        features = features[order]

        # split video and frame
        frame_func = np.vectorize(lambda x: int(str(x)[-8:]))
        vid_func = np.vectorize(lambda x: int(str(x)[:-8]))
        frame_id = frame_func(inds)
        video_id = vid_func(inds)

        total_size = len(inds)
        embed_size = features.shape[1]
        file_path = os.path.join(path, 'extracted_features_{}.hdf5'.format(split))

        f = h5py.File(file_path, 'w')
        # write data
        f.create_dataset("frame_id", (total_size,), dtype='i', data=frame_id)
        f.create_dataset("video_id", (total_size,), dtype='i', data=video_id)
        f.create_dataset("embeddings", (total_size, embed_size), dtype='f', data=features)
        f.create_dataset("targets", (total_size,), dtype='f', data=targets)

        f.close()

    return

def Main(args):
    test_or_val = args.test_set
    cfg_file = os.path.join('configs/config', args.hyper_params)
    cfg = parse_config(cfg_file).config

    print("cfg_file: ", cfg_file)
    ckpt_dir = cfg.CHECKPOINT.DIR
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    #copy2(cfg_file, ckpt_dir)
    num_epochs = cfg.OPTIMIZER.num_epochs

    seed = cfg.SEED_VALUE
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    shuffle = True

    feat_path = os.path.join(
        cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE, "extracted_features_Trunk"
    )
    if not os.path.exists(feat_path): feat_path = feat_path.replace('Trunk', 'Head')
    collect_embeddings(feat_path)
    trainer = LinearEvalTrainer if cfg.MODEL.name == 'FCN' else TeCNOTrainer
    trainer = trainer(cfg)

    train_loader, test_loader, val_loader = create_data_loaders(cfg, shuffle=shuffle)
    if cfg.MODEL.name == 'FCN':
        trainer.add_data_loaders(train_loader, test_loader, val_loader)
        trainer.train()
        trainer.test(data_splits=[test_or_val])
    else:
        trainer.add_data_loaders(train_loader, test_loader, val_loader)
        trainer.train()
        trainer.test(data_splits=['test'])
    return

if __name__ == '__main__':
    parser = create_argument_parser()
    args = parser.parse_args()

    Main(args)
