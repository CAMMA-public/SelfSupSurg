'''
Project: SelfSupSurg
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''

import os
from pathlib import Path

ROOT_DIR = 'slurms/'
header = 'no., status, experiment\n'

def check_status(log_file):
    status = 'RUNNING'
    log = open(log_file, 'r').read()
    if 'All Done!' in log:
        return 'COMPLETED'
    if 'Exception:' in log or 'Error:' in log or 'Traceback' in log:
        return 'ERROR'
    if 'CANCELLED' in log:
        return 'CANCELLED'
    return status

def generate_file(dir):
    print('checking:', dir)
    status_str = header
    status_file = os.path.join(dir, 'status.csv')
    for i, path in enumerate(sorted(Path(dir).rglob('*.log'), key=lambda p: str(p))):
        status = check_status(path)
        status_str += ','.join([str(i), status, str(path)]) + '\n'

    with open(status_file, 'w') as fp:
        fp.write(status_str)

prev_path = []
prev_paths = []
status_str = header
for i, path in enumerate(sorted(Path(ROOT_DIR).rglob('*.log'), key=lambda p: str(p))):
    path_dirs = str(path).split('/')[:-2]
    path_dirs_str = os.path.join(*path_dirs)
    if path_dirs == prev_path:
        continue
    prev_path = path_dirs[:]
    for i, p in enumerate(path_dirs):
        dir = os.path.join(*path_dirs[:i+1])
        if dir in prev_paths:
            continue
        prev_paths.append(dir)
        generate_file(dir)
