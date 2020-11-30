import numpy as np
import pandas as pd
import os, argparse

from load_data_utils import load_exp_annotation, load_keypoints_single_exp
from visualization import (plot_convolutional_syncing_result, 
                           plot_sliding_with_index, 
                           plot_and_save_analysis_fig)
from sync_data_utils import *


__author__ = "Soyong Shin"


def calculate_RMSE(imu, kp):

    return np.sqrt(((imu - kp)**2).mean())


def define_configs(parser):
    
    parser.add_argument('--IMU_dir', type=str, 
                        default='/home/soyongs/dataset/dome_IMU/', 
                        help='IMU data directory')
    parser.add_argument('--KP_dir', type=str, 
                        default='/home/soyongs/dataset/fixed_keypoints/', 
                        help='Keypoints data directory')
    parser.add_argument('--synced-IMU-dir', type=str, 
                        default='/home/soyongs/dataset/syncing_result/', 
                        help='Directory to save synced IMU result')
    parser.add_argument('--save-dir', type=str, 
                        default='/home/soyongs/dataset/processed_data/', 
                        help='Directory to save all loaded file')
    parser.add_argument('--date', type=str, default='190607', help='Date of experiment')
    parser.add_argument('--exp', type=int, default=1, help='n-th of experiment on that date')
    parser.add_argument('--id', type=int, default=1, help='Subject ID (1 or 2)')
    parser.add_argument('--entire-exps', default=False,
                        type=lambda arg: arg.lower() == 'true',
                        help='Check whether syncing entire exps')
    parser.add_argument('--entire-parts', default=False,
                        type=lambda arg: arg.lower() == 'true',
                        help='Check whether syncing entire parts else only sacrum')
    parser.add_argument('--save-video', default=False,
                        type=lambda arg: arg.lower() == 'true',
                        help='Check whether you want to save syncing result as video file')
    parser.add_argument('--save-image', default=True,
                        type=lambda arg: arg.lower() == 'true',
                        help='Check whether you want to save syncing result as image file')

    args = parser.parse_args()

    params = dict()
    params['IMU_dir'] = args.IMU_dir
    params['KP_dir'] = args.KP_dir
    params['synced_IMU_dir'] = args.synced_IMU_dir
    params['save_dir'] = args.save_dir
    params['date'] = args.date
    params['exp'] = args.exp
    params['id'] = args.id
    params['entire_exps'] = args.entire_exps
    params['entire_parts'] = args.entire_parts
    params['save_video'] = args.save_video
    params['save_image'] = args.save_image

    return params


def load_Y_accel_from_keypoints(params, path_kp):
    # keypoints, ids = load_keypoints_single_exp(path_kp, num_file=2000)
    keypoints, ids = load_keypoints_single_exp(path_kp)
    keypoints = keypoints[params['id'] - 1]

    # Calculate acceleration from the variation of sacrum Y value
    accel_, sacrum_height = calculate_accel_from_keypoints(keypoints)
    accel = pd.DataFrame(accel_)[0]
    accel.name = 'KP'
    accel = interpolate_kp_accel(accel)

    # Get peak clue where the hopping occurs
    begin_hop_idx = np.where(sacrum_height == sacrum_height[:1000].min())[0][0]
    begin_hop_idx = 5 * (begin_hop_idx - 1)
    end_hop_idx = np.where(sacrum_height == sacrum_height[-2500:].min())[0][0]
    end_hop_idx = 5 * (end_hop_idx - 1)

    return accel, begin_hop_idx, end_hop_idx


def load_Y_accel_from_IMU(params, path_imu, path_imu_anno):
    accel = pd.read_csv(path_imu, index_col = 'Timestamp (microseconds)')
    accel = accel['Accel Y (g)']
    
    # Load annotation file and get data collection time
    anno_start, anno_end = load_exp_annotation(path_imu_anno)
    
    # Get data collection activities within the date
    i_start_date = np.where(np.array(anno_start > accel.index[0]/1000)==True)[0][0]
    i_end_date = np.where(np.array(anno_end < accel.index[-1]/1000)==True)[0][-1]
    t_start_date = np.array(anno_start)[i_start_date:i_end_date+1]
    t_end_date = np.array(anno_end)[i_start_date:i_end_date+1]

    # Get current experiment data collection activity
    t_start_, t_end_ = t_start_date[params['exp']-1], t_end_date[params['exp']-1]
    idx_start = np.where(accel.index > 1000*t_start_)[0][0]
    idx_end = np.where(accel.index < 1000*t_end_)[0][-1]

    # Temporal segmentation for current experiment
    accel = accel.loc[accel.index[idx_start : idx_end]]
    
    # Drop invalid data (usually data point with accel = -4 or -8)
    invalid_data = (accel == -4) | (accel == -8)
    invalid_index = np.where(np.array(invalid_data) == True)[0]
    invalid_index_ = invalid_data.index[invalid_index]
    accel = accel.drop(invalid_index_)

    # Resample the sensor data with constant fps
    accel = resample_IMU_data(accel)
    
    accel.name = 'IMU'
    
    return accel


def sync_with_sacrum_accel(params):

    print("Syncing data Info: Date (%s), Exp (%d), Subject (%d)"
          %(params['date'], params['exp'], params['id']))

    kp_params, imu_params = refine_params(params)
    
    # Determine the path of data and load acceleration
    path_kp, path_imu_anno, path_imu = get_sacrum_and_kp_path(params)
    imu_accel = load_Y_accel_from_IMU(imu_params, path_imu, path_imu_anno)
    kp_accel, begin_hop, end_hop = load_Y_accel_from_keypoints(kp_params, path_kp)
    
    # Get peaks of each acceleration
    imu_peak = generate_peaks(imu_accel)
    kp_peak = generate_peaks(kp_accel)
    
    # Based on sacrum maximum height at the both ends, generate peaks
    kp_hop = (abs(begin_hop - kp_peak) < 500) | (abs(end_hop - kp_peak) < 500)
    kp_peak = kp_peak[pd.DataFrame(np.where(np.array(kp_hop)==True)[0])[0]]
    
    # Refine IMU and keypoints peaks while the refining does not impact the result
    imu_peak, kp_peak = refine_peaks(imu_accel, kp_accel, imu_peak, kp_peak)

    # Classify front and end (two hopping activities) of keypoints data
    is_front_kp_peak = kp_peak < 5000
    is_front_kp_peak = pd.DataFrame(np.where(np.array(is_front_kp_peak)==True)[0])[0]
    is_end_kp_peak = kp_peak > kp_accel.shape[0] - 5000
    is_end_kp_peak = pd.DataFrame(np.where(np.array(is_end_kp_peak)==True)[0])[0]
    front_kp_peak = kp_peak[kp_peak.index[is_front_kp_peak]]
    end_kp_peak = kp_peak[kp_peak.index[is_end_kp_peak]]
    
    buffer, check_range = 300, 500
    pad = buffer+int(check_range/2)
    
    if front_kp_peak[front_kp_peak.index[0]] < pad:
        buffer, check_range = front_kp_peak[front_kp_peak.index[0]], 200
        pad = buffer+int(check_range/2)

    # Get small window of keypoints accel
    front_kp_idx = [front_kp_peak[front_kp_peak.index[i]] for i in [0, -1]]
    front_kp_idx[-1] = max(front_kp_idx[0] + 300, front_kp_idx[-1])
    front_kp_accel = np.array(kp_accel)[front_kp_idx[0]-buffer:front_kp_idx[1]+buffer]
    
    # Get small window of IMU accel
    front_imu_accel = np.array(imu_accel)[imu_peak[imu_peak.index[0]]-pad:]
    front_imu_accel = front_imu_accel[:front_kp_accel.shape[0] + check_range]
    
    # Slide the front window of KP accel and calculate corresponding RMSEs
    rmse_list = []
    for i in range(check_range):
        start_idx, end_idx = i, i - check_range
        rmse_list += [calculate_RMSE(front_imu_accel[start_idx:end_idx], front_kp_accel)]

    # Calculate frame difference between two accel values
    rmse_result = np.array(rmse_list)
    minimum_idx = np.where(rmse_result == rmse_result.min())[0][0]
    base_diff = imu_peak[imu_peak.index[0]] - front_kp_idx[0]
    diff = base_diff - int(check_range/2) + minimum_idx

    # Check one more time whether synced IMU is correct
    if diff >= 0:
        synced_imu_accel = np.array(imu_accel)[diff:]
    else:
        synced_imu_accel = generate_imu_front_buffer(imu_accel, diff)

    if kp_accel.shape[0] > synced_imu_accel.shape[0]:
        sz_diff = kp_accel.shape[0] - synced_imu_accel.shape[0]
        kp_accel = kp_accel[:kp_accel.index[-sz_diff]]
        print("OpenPose data is longer than IMU data. Need to remove last %d frames of OpenPose data..."\
            %int(sz_diff/5 + 1))
    else:
        synced_imu_accel = synced_imu_accel[:kp_accel.shape[0]]

    front_synced_imu_accel = synced_imu_accel[front_kp_idx[0]-buffer:front_kp_idx[1]+buffer]
    start_idx, end_idx = minimum_idx, minimum_idx - check_range
    check_ = calculate_RMSE(front_synced_imu_accel, front_kp_accel)
    assert rmse_result.min() == check_
    
    # Manually check whether syncing is reasonable
    if params['save_video']:
        plot_convolutional_syncing_result(rmse_result, front_imu_accel, front_kp_accel, check_range)
    
    if params['save_image']:
        if end_kp_peak.shape[0] == 0:
            end_kp_accel = np.array(kp_accel)[-2500:-1000]
            end_synced_imu_accel = synced_imu_accel[-2500:-1000]
        
        else:
            end_idx_ = end_kp_peak.max() + buffer if kp_accel.shape[0] - end_kp_peak.max() > buffer else -1
            start_idx_ = max(end_kp_peak.min(), end_idx_ - 1000)
            end_kp_accel = np.array(kp_accel)[start_idx_:end_idx_]
            end_synced_imu_accel = synced_imu_accel[start_idx_:end_idx_]
        
        plot_and_save_analysis_fig(params, front_synced_imu_accel, front_kp_accel,
                                   end_synced_imu_accel, end_kp_accel)

    
    synced_imu_index = imu_accel.index[diff:]
    synced_imu_index = synced_imu_index[:kp_accel.shape[0]]
    
    return synced_imu_index


def sync_all_sensors_from_sacrum(params):
    
    # 15 sensors list
    sensor_list = ["chest", "head", "lbicep", "lfoot", "lforearm", "lhand", "lshank", "lthigh",
                   "rbicep", "rfoot", "rforearm", "rhand", "rshank", "rthigh", "sacrum"]
    
    # Get syncing index (timestamp) from sacrum accel Y
    synced_imu_index = sync_with_sacrum_accel(params)
    
    for part in sensor_list:
        current_fldr, target_fldr = get_imu_fldr(params, part)
        for sensor in ['accel', 'gyro']:
            data = pd.read_csv(os.path.join(current_fldr, sensor+'.csv'), 
                               index_col = 'Timestamp (microseconds)')
            synced_imu_index_ = delete_overlap_index(data.index, synced_imu_index)
            empty_data = pd.DataFrame(index=synced_imu_index_, columns=data.columns)
            data = data.append(empty_data).sort_index(axis=0)
            data = data.interpolate(limit_area='inside')
            data = data.loc[synced_imu_index]
            data.to_csv(os.path.join(target_fldr, '%s%02d.csv'%(sensor, params['exp'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    params = define_configs(parser)
    
    if params['entire_parts']:
        sync_function = sync_all_sensors_from_sacrum
    else:
        sync_function = sync_with_sacrum_accel

    if params['entire_exps']:
        date_list = ['190503', '190510', '190517', '190607']
        id_list = [1, 2]
        for id in id_list:
            params['id'] = id
            for date in date_list:
                params['date'] = date
                path_kp = '/home/soyongs/dataset/fixed_keypoints/'
                path_kp = os.path.join(path_kp, date)
                _, fldrs, _ = next(os.walk(path_kp))
                num_exps = len(fldrs)
                for exp in range(num_exps):
                    params['exp'] = exp + 1
                    # TODO: Pass the syncing if the file exists
                    if not params['entire_parts']:
                        img_file = 'output/sync_analysis/Set%02d/%s/exp%02d.png'%(id, date, exp+1)
                        if os.path.isfile(img_file):
                            print("Syncing result already exists at %s! Skipping this sequence ..."%img_file)
                            continue
                    
                        kp, _ = refine_params(params)
                        if kp['id'] == 100:
                            print("There is only one subject in this experiment (No Subject ID %d here)"%id)
                            continue
                    
                    # TODO: Deal with two experiments where only one subject exists
                    sync_function(params)
    else:
        sync_function(params)