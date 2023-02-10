'''
Project: SelfSupSurg
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''

import os
import json
from pathlib import Path

ROOT_DIR = 'runs/'
header = 'no., epoch, test, train, experiment\n'

def get_metric_type(type='phase'):
    metric = 'top_1'
    if type == 'tool':
        metric = 'mAP'
    return metric

def get_results(file, type='phase'):
    try:
        with open(file) as f:
            metrics = [json.loads(line) for line in f.readlines()]

        metric = get_metric_type(type)
        results = [
                    metrics[-1]['train_phase_idx'],
                    metrics[-1]['test_accuracy_list_meter'][metric]['0'],
                    metrics[-2]['train_accuracy_list_meter'][metric]['0']
                ]
    except:
        results = [-1, 0.00, 0.00]
    return results

def generate_file(dir, type='phase'):
    print('collating folder:', dir)
    results_str = header
    results_file = os.path.join(dir, 'results_{:s}.csv'.format(type))
    for i, path in enumerate(sorted(Path(dir).rglob('metrics.json'), key=lambda p: str(p))):
        if 'test' not in str(path) and type not in str(path):
            continue

        # experiment = os.path.join(*str(path).split('/')[1:-2])
        experiment = os.path.join(*str(path).split(str(dir))[:])
        results = ','.join(map(lambda v: "{:.2f}".format(v),get_results(path)))
        results_str += ','.join([str(i), results, experiment]) + '\n'

    try:
        with open(results_file, 'w') as fp:
            fp.write(results_str)
    except:
        print('warning: could not write to folder:', dir)

prev_path = []
prev_paths = []
types = ['phase']
results_str = header

for i, path in enumerate(sorted(Path(ROOT_DIR).rglob('metrics.json'), key=lambda p: str(p))):
    if 'test' not in str(path):
        continue

    path_dirs = str(path).split('/')[:-3]
    path_dirs_str = os.path.join(*path_dirs)
    if path_dirs == prev_path:
        continue
    prev_path = path_dirs
    for i, p in enumerate(path_dirs[:1]):
        dir = os.path.join(*path_dirs[:i+1])
        if dir in prev_paths:
            continue
        prev_paths.append(dir)
        for type in types:
            generate_file(dir, type=type)
