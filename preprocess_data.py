import numpy as np
import torch
import pickle
import os


__author__ = "Soyong Shin"


def to_pkl_file(data, data_name='keypoints'):
    assert data_name in ['keypoints', 'accel', 'gyro']
    
    path_save = '/home/soyongs/dataset/processed_data/'
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    
    with open(os.path.join(path_save, '%s.pkl'%data_name), 'wb') as fopen:
        pickle.dump(data, fopen)
    
    pass


def to_torch_tensor(data):

    data_ = []
    for d in data:
        d_ = torch.tensor(d)
        if torch.cuda.is_available():
            d_ = d_.cuda()
        
        data_.append(d_.float())
    
    return data_