import numpy as np
import pandas as pd
import os
from os.path import join

from load_data_utils import *
from preprocess_data import *
from sync_data_utils import refine_params, get_sacrum_and_kp_path


__author__ = "Soyong Shin"


def load_imu_single_exp(params):
    
    path = '/home/soyongs/dataset/syncing_result/'
    path = join(path, 'Set%02d'%params['id'], params['date'])
    sensor_list = ["chest", "head", "lbicep", "lfoot", "lforearm", "lhand", "lshank", "lthigh",
                   "rbicep", "rfoot", "rforearm", "rhand", "rshank", "rthigh", "sacrum"]
    
    for sensor in sensor_list:
        tmp_accel = pd.read_csv(join(path, sensor, 'accel%02d.csv'%params['exp']))
        tmp_gyro = pd.read_csv(join(path, sensor, 'gyro%02d.csv'%params['exp']))
        tmp_accel = np.array(tmp_accel)[:, None, 1:]
        tmp_gyro = np.array(tmp_gyro)[:, None, 1:]
        
        if sensor == 'chest':
            accel, gyro = np.array(tmp_accel), np.array(tmp_gyro)
        else:
            accel = np.concatenate((accel, tmp_accel), axis=1)
            gyro = np.concatenate((gyro, tmp_gyro), axis=1)

    return accel, gyro


def generate_keypoints_pkl_file():
    params = dict()

    date_list = ['190503', '190510', '190517', '190607']
    id_list = [1, 2]
    keypoints = []
    for id in id_list:
        params['id'] = id
        for date in date_list:
            params['date'] = date
            path_kp = '/home/soyongs/dataset/fixed_keypoints/'
            path_kp = join(path_kp, date)
            _, fldrs, _ = next(os.walk(path_kp))
            num_exps = len(fldrs)
            for exp in range(num_exps):
                params['exp'] = exp + 1
                kp_params, _ = refine_params(params)
                path_kp_, _, _ = get_sacrum_and_kp_path(kp_params)
                keypoints_, _ = load_keypoints_single_exp(path_kp_)                
                try:
                    keypoints_ = keypoints_[kp_params['id'] - 1]
                    print("Keypoints loading info : subject(%d) date(%s) exp(%d)"
                          %(params['id'], params['date'], params['exp']))
                    keypoints.append(keypoints_)
                except:
                    print("Not feasible info : subject(%d) date(%s) exp(%d)"
                          %(params['id'], params['date'], params['exp']))
                    continue

    to_pkl_file(to_torch_tensor(keypoints), 'keypoints')


def generate_imu_pkl_file():
    params = dict()
    
    date_list = ['190503', '190510', '190517', '190607']
    id_list = [1, 2]
    accel, gyro = [], []
    for id in id_list:
        params['id'] = id
        for date in date_list:
            params['date'] = date
            path_kp = '/home/soyongs/dataset/fixed_keypoints/'
            path_kp = join(path_kp, date)
            _, fldrs, _ = next(os.walk(path_kp))
            num_exps = len(fldrs)
            for exp in range(num_exps):
                params['exp'] = exp + 1
                try:
                    accel_, gyro_ = load_imu_single_exp(params)
                    print('IMU sensor loading info : subject(%d) date(%s) exp(%d)'
                          %(id, date, exp+1))
                    accel.append(accel_)
                    gyro.append(gyro_)
                except:
                    print("Not feasible info : subject(%d) date(%s) exp(%d)"
                          %(params['id'], params['date'], params['exp']))

    to_pkl_file(to_torch_tensor(accel), 'accel')
    to_pkl_file(to_torch_tensor(gyro), 'gyro')


if __name__ == "__main__":
    generate_imu_pkl_file()
    generate_keypoints_pkl_file()