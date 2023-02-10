'''
Project: SelfSupSurg
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''


import os
import sys
import glob
import json
import numpy as np
from pathlib import Path
from skimage import measure

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score

from model_helpers import mAP

root_dir = 'runs/'
#sub_dir = 'ssl_ablation'
#sub_dir = 'ssl_main/transferability/'
#sub_dir = 'ssl_main/finetuning/cholec_to_cholec'
#sub_dir = 'ssl_main/bypass40/'
#sub_dir = "ssl_ablation/fps/series_01" runs/ssl_main_v2/finetuning/cholec_to_cholec
sub_dir = 'ssl_main_v2/finetuning/'
feat_dir = 'extracted_features'
header_tool = 'no, mAP, experiment\n'
header_phase = 'no, accuracy, precision, recall, f-score, support, experiment\n'
header_tool_class = 'no, AP_T1, AP_T2, AP_T3, AP_T4, AP_T5, AP_T6, AP_T7, experiment\n'
header_phase_class = 'no, F1_P1, F1_P2, F1_P3, F1_P4, F1_P5, F1_P6, F1_P7, experiment\n'

def class_metrics(labels, predictions, num_cls=7):
    exp_labels = np.array(range(num_cls))
    missing = [idx for idx in exp_labels if idx not in labels and idx not in predictions]
    class_score = score(labels, predictions)
    for miss in missing:
        class_score = [np.insert(np.float32(sc), miss, np.nan) for sc in class_score]

    return class_score

def read_predictions(path):
    path_str = str(path)
    if os.path.isfile(os.path.join(path_str,'inds_all.npy')):
        inds = np.load(os.path.join(path_str,'inds_all.npy'))
        preds = np.load(os.path.join(path_str,'features_all.npy'))
        targets = np.load(os.path.join(path_str,'targets_all.npy'))
    else:
        finds = sorted(glob.glob(os.path.join(path_str,'*inds.npy')))
        fpreds = sorted(glob.glob(os.path.join(path_str,'*features.npy')))
        ftargets = sorted(glob.glob(os.path.join(path_str,'*targets.npy')))
        if not finds: return [], [], []
        inds = np.concatenate([np.load(f) for f in finds])
        preds = np.concatenate([np.load(f) for f in fpreds])
        targets = np.concatenate([np.load(f) for f in ftargets])
        nps = [inds, preds, targets]
        fnames = ['inds', 'features', 'targets']
        for i, arr in enumerate(nps):
            with open(os.path.join(path_str, '{}_all.npy'.format(fnames[i])), 'wb') as f:
                np.save(f, arr)
        files = finds + fpreds + ftargets
        for f in files:
            os.remove(f)

    idxs = np.argsort(inds)
    inds = inds[idxs]
    preds = preds[idxs]
    targets = targets[idxs]

    return inds, preds, targets

def normalize_predictions(predicts):
    predicts_norm = np.argmax(predicts, axis=1)
    return predicts_norm

def compute_phase_scores(inds, labels, predicts, agg, directory=''):
    if agg == 'class' and len(labels) == 0: return [-1] * 7,[]
    if len(labels) == 0: return [-1] * 5, []
    labels = labels.squeeze()
    preds = normalize_predictions(predicts)
    if agg == 'frame':
        scores = score(labels, preds)

        acc = np.sum(labels == preds) * 100 / len(labels)
        acc = np.around(acc, 2)

        mean = np.mean(np.vstack(scores).T, axis=0)
        mean[:-1] *= 100
        mean = np.around(mean, 2)
        mean = [acc] + mean.tolist()

        std = np.std(np.vstack(scores).T, axis=0)
        std[:-1] *= 100
        std = np.around(std, 2)
        std = std.tolist()

    elif agg == 'class':
        # scores = score(labels, preds)
        # mean = (np.array(scores[2]) * 100).tolist()
        # std = np.zeros_like(mean).tolist()
        vid = np.floor_divide(inds, 100000000)
        #print('videos:', np.unique(vid))
        class_f1 = []
        for v in np.unique(vid):
            sub_inds = np.argwhere(vid == v)
            sub_labels = labels[sub_inds]
            sub_preds = preds[sub_inds]

            # compute F1
            vid_score = class_metrics(sub_labels, sub_preds)
            class_f1.append(np.array(vid_score[2])*100)

        print(len(class_f1), len(class_f1[0]))
        mean = np.around(np.nanmean(class_f1, axis=0), 2).tolist()
        std = np.around(np.nanstd(class_f1, axis=0), 2).tolist()
    elif agg == 'video':
        # split labels, preds by video
        vid = np.floor_divide(inds, 100000000)
        #print('videos:', np.unique(vid))
        accs = []
        scores = []
        for v in np.unique(vid):
            sub_inds = np.argwhere(vid == v)
            sub_labels = labels[sub_inds]
            sub_preds = preds[sub_inds]

            # compute acc and append
            vid_acc = np.sum(sub_labels == sub_preds) * 100 / len(sub_labels)
            accs.append(vid_acc)

            # compute F1
            vid_score = score(sub_labels, sub_preds)
            mean = np.mean(np.vstack(vid_score).T, axis=0)
            mean[:-1] *= 100
            scores.append(mean)

        # summarize
        overall_acc = np.mean(np.stack(accs))
        overall_acc = np.around(overall_acc, 2)

        overall_f1 = np.mean(np.stack(scores), axis=0)
        overall_f1 = np.around(overall_f1, 2)

        mean = [overall_acc] + overall_f1.tolist()

        std = np.std(np.stack(scores), axis=0)
        std = np.around(std, 2)
        std = [np.std(np.stack(accs))] + std.tolist()

    elif agg == 'video_relaxed':
        # split labels, preds by video
        frame_order = np.argsort(inds)
        labels = labels[frame_order]
        preds = preds[frame_order]
        inds = inds[frame_order]
        vid = np.floor_divide(inds, 100000000)
        accs = []
        scores = []
        for v in np.unique(vid):
            sub_inds = np.argwhere(vid == v)
            sub_labels = labels[sub_inds]
            sub_preds = preds[sub_inds]

            vid_prec, vid_rec, vid_f1, vid_jacc, vid_acc = compute_phase_relaxed_scores(sub_preds,
                    sub_labels)
            accs.append(vid_acc)
            scores.append([np.nanmean(vid_prec), np.nanmean(vid_rec), np.nanmean(vid_f1), -1])

        mean = [np.mean(np.stack(accs))] + np.mean(np.stack(scores), axis=0).tolist()
        std = [np.std(np.stack(accs))] + np.std(np.stack(scores), axis=0).tolist()

    return mean, std

def compute_phase_relaxed_scores(preds, targets, boundary_size=10):
    #EVALUATE
    # A function to evaluate the performance of the phase recognition method
    # providing jaccard index, precision, and recall for each phase 
    # and accuracy over the surgery. All metrics are computed in a relaxed
    # boundary mode.
    # OUTPUT:
    #    res: the jaccard index per phase (relaxed) - NaN for non existing phase in GT
    #    prec: precision per phase (relaxed)        - NaN for non existing phase in GT
    #    rec: recall per phase (relaxed)            - NaN for non existing phase in GT
    #    acc: the accuracy over the video (relaxed)
    res, prec, rec = [], [], []
    diff = preds - targets
    updatedDiff = diff.copy()

    # obtain the true positive with relaxed boundary
    for iPhase in range(7):
        labels, num = measure.label(targets == iPhase, return_num=True)

        for iConn in range(1, num + 1):
            comp = np.argwhere(labels == iConn)
            startIdx = np.min(comp)
            endIdx = np.max(comp) + 1

            curDiff = diff[startIdx:endIdx]

            # in the case where the phase is shorter than the relaxed boundary
            t = boundary_size
            if t > len(curDiff):
                t = len(curDiff)

            # relaxed boundary
            # revised for cholec80 dataset !!!!!!!!!!!
            if iPhase == 3 or iPhase == 4: # Gallbladder dissection and packaging might jump between two phases
                curDiff[:t][curDiff[:t] == -1] = 0 # late transition

                # early transition, 5 can be predicted as 6/7 at the end > 5 followed by 6/7
                curDiff[-t:][curDiff[-t:] == 1] = 0
                curDiff[-t:][curDiff[-t:] == 2] = 0

            elif iPhase == 5 or iPhase == 6: # Gallbladder dissection might jump between two phases
                # late transition
                curDiff[:t][curDiff[:t] == -1] = 0
                curDiff[:t][curDiff[:t] == -2] = 0

                # early transition
                curDiff[-t:][curDiff[-t:] == 1] = 0
                curDiff[-t:][curDiff[-t:] == 2] = 0

            else:
                # general situation
                curDiff[:t][curDiff[:t] == -1] = 0 # late transition
                curDiff[-t:][curDiff[-t:] == 1] = 0 # early transition

            updatedDiff[startIdx:endIdx] = curDiff

    # compute jaccard index, prec, and rec per phase
    for iPhase in range(7):
        gt_num = (targets == iPhase).sum()
        if gt_num == 0:
            # no iPhase in current ground truth, assigned NaN values
            # SHOULD be excluded in the computation of mean (use nanmean)
            res.append(np.nan)
            prec.append(np.nan)
            rec.append(np.nan)
            continue

        # get all indices where pred is iPhase
        tp_and_fp = np.argwhere(preds == iPhase).flatten()
        tp_and_fn = np.argwhere(targets == iPhase).flatten()
        union = np.union1d(tp_and_fp, tp_and_fn)

        # compute tp
        tp = np.sum(updatedDiff[tp_and_fp] == 0)

        # divide by union to get jaccard
        jaccard = tp / len(union)
        jaccard = jaccard * 100

        res.append(jaccard)

        # Compute prec and rec
        prec.append(tp * 100 / len(tp_and_fp))
        rec.append(tp * 100 / len(tp_and_fn))

    # compute accuracy
    acc = sum(updatedDiff == 0) / len(targets)
    acc = acc * 100

    # compute f1
    prec = np.array(prec)
    rec = np.array(rec)
    f1 = 2 * prec * rec / (prec + rec)
    res = np.array(res)

    return prec, rec, f1, res, acc

def compute_tool_scores(inds, labels, predicts, agg, directory):
    print('dir: ', directory)
    try:
        mean = [mAP(labels, predicts, istensor=False) * 100]
        std = [0.00]
        if agg == 'class':
            mean = mAP(labels, predicts, mean=False, istensor=False) * 100
            std = [0.00] * 7
        #elif agg == 'video':
        #    # split labels, preds by video
        #    vid = np.floor_divide(inds, 100000000)
        #    accs = []
        #    scores = []
        #    for v in np.unique(vid):
        #        sub_inds = np.argwhere(vid == v)
        #        sub_labels = labels[sub_inds]
        #        sub_preds = preds[sub_inds]

        #        # compute mAP
        #        vid_score = score(sub_labels, sub_preds)
        #        mean = np.mean(np.vstack(vid_score).T, axis=0)
        #        mean[:-1] *= 100
        #        scores.append(mean)

        #    # summarize
        #    overall_mIOU = np.mean(np.stack(scores, axis=0))
        #    overall_mIOU = np.around(overall_mIOU, 2)
        #    mean = overall_mIOU

        #    std = np.std(np.stack(scores), axis=0)
        #    std = np.around(std, 2)
        #    std = std.tolist()
    except:
        mean = [-1] * 7 if agg == 'class' else [-1.00]
        std = [0.00] * 7 if agg == 'class' else [0.00]

    return mean, std

def collect_metrics(directory, task='phase', agg='frame'):
    inds, preds, targets = read_predictions(directory)
    score_fn = compute_tool_scores if task == 'tools' else compute_phase_scores
    metrics, stds = score_fn(inds, targets, preds, agg, directory)
    return metrics, stds

def metrics_collator(directory, agg, task='phase', name='sanat'):
    results_str = header_tool if task =='tools' else header_phase
    results_str = header_tool_class if agg == 'class' and task =='tools' else header_phase_class if agg == 'class' and task =='phase' else results_str 
    results_file = os.path.join(directory, 'metrics_{:s}.csv'.format('_'.join([task, name])))
    new_dir = os.path.join(directory, sub_dir)
    print('search: ', directory)
    for i, path in enumerate(sorted(Path(new_dir).rglob(feat_dir), key=lambda p: str(p))):
        if i == 0: print('collating dir:', directory)
        if task not in str(path): continue
        if feat_dir+'_Trunk' in str(path): continue
        experiment = os.path.join(*str(path).split(str(directory))[-1].split('/')[:-1])
        metric_and_std = collect_metrics(path, task, agg)
        #print(metric_and_std)
        results = ','.join(map(lambda x, y: "{:.2f} +- {:.2f}".format(x, y), *metric_and_std))
        results_str += ','.join([str(i), results, experiment]) + '\n'

    if results_str == header_tool or results_str == header_phase or \
        results_str == header_phase_class or results_str == header_tool_class: return

    try:
        with open(results_file, 'w') as fp:
            fp.write(results_str)
        print('collating dir: done!!!')
    except:
        print('warning: could not write to folder:', directory)
    return

def Main(args, agg):
    prev_path = []
    prev_paths = []
    tasks = ['phase', 'tools']
    name = '_'.join(args)
    for task in tasks:
        prev_path = []
        prev_paths = []
        print('Evaluation task:', task)
        for i, path in enumerate(sorted(Path(root_dir).rglob(task), key=lambda p: str(p))):
            print('test:', path)
            if 'test' not in str(path):
                print('cont')
                continue
            path_dirs = str(path).split('/')
            path_dirs_str = os.path.join(*path_dirs)
            if path_dirs == prev_path:
                continue
            prev_path = path_dirs
            for i, p in enumerate(path_dirs[:1]):
                src_dir = os.path.join(*path_dirs[:i+1])
                if src_dir in prev_paths:
                    continue

                prev_paths.append(src_dir)
                metrics_collator(src_dir, agg, task=task, name=name)

    return

if __name__ == '__main__':
    if len(sys.argv) <= 2:
        print('Usage: python generate_test_results.py <user name> <hpc/jz>')
        sys.exit(-1)

    if sys.argv[2] not in ['hpc', 'jz']:
        print('Wrong server name!!!')
        print('Usage: python generate_test_results.py <user name> <hpc/jz>')
        sys.exit(-1)

    if len(sys.argv) > 3:
        agg_mode = sys.argv[-1]
    else:
        agg_mode = 'video'

    Main(sys.argv[1:3], agg_mode)
