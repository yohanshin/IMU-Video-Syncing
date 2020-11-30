import numpy as np
import pandas as pd
import os, sys
import csv, json

from tqdm import tqdm


__author__ = "Soyong Shin"


"""
Utility functions for Dome experiments data loading.
"""

def load_exp_annotation(filename):
    """
    Input : File name
    Output : Start and End time of each experiment data collection
    """
    
    annotation = pd.read_csv(filename, index_col='AnnotationId')
    t_start = annotation['Start Timestamp (ms)']['Activity:Data Collection']
    t_end = annotation['Stop Timestamp (ms)']['Activity:Data Collection']
    
    return t_start, t_end


def load_json_single_frame(file_name):
    """
    Input : File name
    Output : index, 26 joints location
    """
    ids, joints = [[], []]
    with open(file_name) as json_file:
        data = json.load(json_file)
        for body in data["bodies"]:
            ids.append(body['id'])
            joints.append(np.array(body['joints26']).reshape(1,-1,4))
    try: 
        joints = np.vstack(joints)
    except:
        pass

    return ids, joints


def load_keypoints_single_exp(path, num_file=-1):
    """
    Load keypoints data from single experiment (two subjects)
    Input : Experiment path
    Output : index, 26 joints location for entire frames
    """
    _, _, files_ = next(os.walk(path))
    files_.sort()
    joints = []
    n_file = len(files_) if num_file == -1 else num_file
    with tqdm(total=n_file, leave=True) as prog_bar:
        for file_ in files_[:n_file]:
            ids, cur_joints = load_json_single_frame(os.path.join(path, file_))
            for id in ids:
                if len(joints) == 0:
                    joints.append(cur_joints[0][None])
                elif len(joints) == 1 and len(cur_joints) == 2:
                    joints.append(cur_joints[1][None])
                else:
                    joints[id] = np.vstack([joints[id], cur_joints[id][None]])
            prog_bar.update(1)
            prog_bar.refresh()
    
    ids = [i for i in range(len(joints))]
    
    # Some experiments record openpose first
    if path.split('/')[-2] == '190607' and path[-2:] == '13':
        joints[0] = joints[0][150:]
        joints[1] = joints[1][150:]
    
    return joints, ids
    