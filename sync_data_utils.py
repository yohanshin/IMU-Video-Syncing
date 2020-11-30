import numpy as np
import pandas as pd
import os
from copy import copy


__author__ = "Soyong Shin"


def refine_params(params):
    """
    Some experiments were conducted with only one subject
    Refining parameters (Experiments & Subject ID) after the experiments
    Subject ID  --> for loading keypoints
    Experiments --> for loading IMU
    """

    kp_params, imu_params = copy(params), copy(params)
    date, exp, id = params['date'], params['exp'], params['id']
    if date == '190517':
        if exp == 12:
            id = 100 if id == 1 else 1
    elif date == '190607':
        if exp == 11:
            id = 100 if id == 2 else 1
        if exp > 11 and id == 2:
            exp = exp - 1
    
    kp_params['id'] = id
    imu_params['exp'] = exp
    
    return kp_params, imu_params


def get_sacrum_and_kp_path(params):
    
    base_path_imu = params['IMU_dir']
    path_imu_anno = os.path.join(base_path_imu, 'Set%02d'%params['id'], 'Set%02d_annotations.csv'%params['id'])
    path_imu = os.path.join(base_path_imu, 'Set%02d'%params['id'], params['date'], 'sacrum', 'accel.csv')
    
    base_path_kp = params['KP_dir']
    path_kp = os.path.join(base_path_kp, params['date'], 'exp%02d'%params['exp'])
    
    return path_kp, path_imu_anno, path_imu


def get_imu_fldr(params, part):
    
    current_fldr = params['IMU_dir']
    current_fldr = os.path.join(current_fldr, 'Set%02d'%params['id'], params['date'], part)

    target_fldr = params['synced_IMU_dir']
    target_fldr = os.path.join(target_fldr, 'Set%02d'%params['id'], params['date'], part)
    if not os.path.exists(target_fldr):
        os.makedirs(target_fldr)
    
    return current_fldr, target_fldr


def interpolate_kp_accel(kp_accel):

    # Generate empty data with dense index
    dense_index = [i/5 for i in range(5 * kp_accel.index[-1]) if i%5 != 0]
    dense_index = pd.Index(dense_index)
    dense_index.name = 'KP'
    empty_data = pd.Series(index=dense_index)
    
    # Interpolate with dense index
    kp_accel = kp_accel.append(empty_data).sort_index(axis=0)
    kp_accel = kp_accel.interpolate(limit_area='inside')
    
    return kp_accel


def resample_IMU_data(accel):
    """
    MC10 IMU sensor has variational (albeit very small) frame-per-second (FPS).
    Assuming that camera has a constant FPS, resample MC10 IMU data to a constant FPS.
    """

    # Generate empty data with constant FPS
    trg_period = 8000  # FPS = 125 : 8000ms/data
    trg_num_data = int((accel.index[-1] - accel.index[0])/trg_period)
    resample_index = [accel.index[0] + 8000*(i+1) for i in range(trg_num_data)]
    resample_index = pd.Index(resample_index)
    un_overlaped_index = delete_overlap_index(accel.index, resample_index)
    empty_data = pd.Series(index=un_overlaped_index)

    # INterpolate at constant data FPS points
    accel = accel.append(empty_data).sort_index(axis=0)
    accel = accel.interpolate(limit_area='inside')

    # Extract data with resampled index
    accel = accel[resample_index]

    return accel


def generate_peaks(data, thresh=2):
    """
    Generate the local peak points (local maximum) larger than threshold
    Data should be instance of pandas.DataFrame
    """

    is_peak = (data.shift(-1) < data) & (data.shift(1) < data) & (data > thresh)
    peak_idx = pd.DataFrame(np.where(np.array(is_peak)==True)[0])[0]
    return peak_idx


def refine_peaks_one_step(accel, index):
    """
    From detected peak points, drop some of them are not seemed to be a hopping sequence.
    1) Drop isolated peak points (Because hopping happens three to four times in few seconds)
    2) If two peaks are very close, choose the bigger peaks.
    3) Drop the mid-peaks induced by other activities
    """

    max_gap, min_gap = 125, 35

    if len(index) > 20:
        index = index.drop(index.index[10:-10])
    isolated = ((index - index.shift(1)) > max_gap) & ((index.shift(-1) - index) > max_gap)
    non_isolated = isolated == False
    index = index[non_isolated]

    bfl = index.index
    output = copy(index)
    for i in range(index.shape[0]):
        if i == 0 and index[bfl[1]] - index[bfl[0]] > max_gap:
            output = output.drop(bfl[i])
        elif i == bfl.shape[0]-1 and (index[bfl[i]] - index[bfl[i-1]]) > max_gap:
            output = output.drop(bfl[i])
        elif i > 0 and (index[bfl[i]] - index[bfl[i-1]]) < min_gap:
            if accel[accel.index[index[bfl[i]]]] > accel[accel.index[index[bfl[i-1]]]]:
                try:
                    output = output.drop(bfl[i-1])
                except:
                    continue
            else:
                output = output.drop(bfl[i])
    return output


def refine_peaks(imu_accel, kp_accel, imu_peak, kp_peak):
    
    imu_peak_, kp_peak_ = [], []
    
    while len(imu_peak_) != len(imu_peak):
        if len(imu_peak_) != 0:
            imu_peak = imu_peak_
        imu_peak_ = refine_peaks_one_step(imu_accel, imu_peak)
    
    while len(kp_peak_) != len(kp_peak):
        if len(kp_peak_) != 0:
            kp_peak = kp_peak_
        kp_peak_ = refine_peaks_one_step(kp_accel, kp_peak)

    return imu_peak, kp_peak


def delete_overlap_index(d_index, s_index):
    """
    Check and delete overlap index which exists in both raw and synced data
    """
    
    sampling_period = 8000
    o_index = d_index[d_index - s_index[0] > (-1)*sampling_period]
    o_index = o_index[o_index - s_index[-1] < sampling_period]
    o_index = o_index[(o_index - s_index[0]) % sampling_period == 0]
    s_index = s_index.drop(o_index)

    return s_index


def generate_imu_front_buffer(accel, diff):
    
    sampling_period = 8000
    new_index = np.arange(diff, 0) * sampling_period + accel.index[0]
    new_index = pd.Index(new_index)
    empty_data = pd.Series(index=new_index)
    accel = accel.append(empty_data).sort_index(axis=0)
    accel = np.array(accel)
    
    return accel


def refine_accel_with_zero_conf(accel, conf, target_value=1):
    """
    Change the acceleration value where sacrum is not detected to target value (1)
    """
    
    is_not_zero = conf > 0
    check = (is_not_zero[:-2] * is_not_zero[1:-1] * is_not_zero[2:]) == 0
    accel[check] = target_value

    return accel


def calculate_accel_from_keypoints(keypoints, idx=2):
    """
    Calculate acceleration from the variation of the part (sacrum) Y value
    """

    height = keypoints[:, idx, 1]
    accel = (-1) * (height[:-2] + height[2:] - 2 * height[1:-1])
    accel = accel * 25 * 25 / 100 / 9.81 + 1
    
    conf = keypoints[:, -2, -1]
    accel = refine_accel_with_zero_conf(accel, conf)
    
    accel_ = np.ones(accel.shape[0] + 2)
    accel_[1:-1] = accel

    return accel, height