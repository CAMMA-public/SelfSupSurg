"""
dataloader for CholecT50 dataset
"""

import os

import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .dataloader import CholecT50
import ivtmetrics

import logging


def evaluate(net, args, records, mode="test", ignore_null=False):
    rec = ivtmetrics.Recognition(100)
    rec.reset_global()

    # print(f"Starting evaluation for {mode} data")
    results_vids = []
    for vid in tqdm(records, desc="overall progress", colour="green"):
        dataloader   = CholecT50(args.DATA.HOME, vid, split=mode)(batch_size=args.EVAL.BS, num_workers=args.EVAL.NW, shuffle=False)
        size = len(dataloader.dataset)

        if mode == 'test': # or 'validation':
            print(f"Processing video {vid} of length: {size}")
        
        #loader = tqdm(dataloader, unit="batch")
        with torch.no_grad():       
            for i, (frames, (y_i, y_v, y_t, y_ivt)) in enumerate(tqdm(dataloader, desc="single video progress", colour="blue")): 
                net.eval()
                y_ivt  = y_ivt.squeeze(1).cuda()
                frames = frames.cuda()   
                
                logit_ivt = net(frames)
                rec.update(y_ivt.float().detach().cpu(), torch.sigmoid(logit_ivt).detach().cpu())

            ap_i   = rec.compute_AP('i', ignore_null=ignore_null)['mAP']
            ap_v   = rec.compute_AP('v', ignore_null=ignore_null)['mAP']
            ap_t   = rec.compute_AP('t', ignore_null=ignore_null)['mAP']
            ap_ivt = rec.compute_AP('ivt', ignore_null=ignore_null)['mAP']
            results_vids.append({"vid": vid, "ap_i": ap_i, "ap_v": ap_v, "ap_t": ap_t, "ap_ivt": ap_ivt})
            rec.video_end()
        logging.info(f'\nVideo # {str(vid).zfill(3)}: AP_i = {ap_i:.4f} | AP_v = {ap_v:.4f} | AP_t = {ap_t:.4f} | AP_ivt = {ap_ivt:.4f}')
        
    logging.info(f'Computing the final metrics')
    # compute the final mAP for all the test videos
    imAP   = rec.compute_video_AP('i', ignore_null=ignore_null)['mAP']
    vmAP   = rec.compute_video_AP('v', ignore_null=ignore_null)['mAP']
    tmAP   = rec.compute_video_AP('t', ignore_null=ignore_null)['mAP']
    ivtmAP = rec.compute_video_AP('ivt', ignore_null=ignore_null)['mAP']
    ivmAP  = rec.compute_video_AP('iv', ignore_null=ignore_null)['mAP']
    itmAP  = rec.compute_video_AP('it', ignore_null=ignore_null)['mAP']

    # topk values
    itopk   = rec.topK(args.EVAL.TOPK, 'i')
    ttopk   = rec.topK(args.EVAL.TOPK, 't')
    vtopk   = rec.topK(args.EVAL.TOPK, 'v')
    ivttopk = rec.topK(args.EVAL.TOPK, 'ivt')
    logging.info('Eval RDV split:     AP_i  |  AP_v  |  AP_t  | AP_ivt')
    logging.info('-'*52) 
    for _r in results_vids:
        logging.info(f'Video # {str(_r["vid"]).zfill(3)} AP : {_r["ap_i"]:.4f} | {_r["ap_v"]:.4f} | {_r["ap_t"]:.4f} | {_r["ap_ivt"]:.4f}')

    logging.info(f'MODE: {mode} || mAP  ==> i: {imAP:.3f} || v: {vmAP:.3f} || t: {tmAP:.3f} || ivt: {ivtmAP:.3f}  iv: {ivmAP:.3f} it: {itmAP:.3f}---- |')
    logging.info(f'MODE: {mode} || topK ==> tool: {itopk:.3f} || verb: {vtopk:.3f} || target: {ttopk:.3f} || triplet: {ivttopk:.3f} ----')

    return { 
             'tool_mAP':imAP,
             'verb_mAP':vmAP,
             'target_mAP':tmAP,
             'triplet_mAP':ivtmAP
        }