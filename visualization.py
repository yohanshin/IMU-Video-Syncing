import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import pandas as pd
import cv2, os
from tqdm import tqdm


__author__ = "Soyong Shin"


def random_rotate_in_y_axis(keypoints):
    rot = np.random.random(1)[0] * 2 * np.pi
    x, y, z, conf = np.split(keypoints, 4, axis=1)
    x_ = x * np.cos(rot) + z * np.sin(rot)
    z_ = -x * np.sin(rot) + z * np.cos(rot)

    keypoints[:, 0] = x_[:, 0]
    keypoints[:, 2] = z_[:, 0]
    
    return keypoints


def rotate_to_face_front(keypoints):
    
    lpelvis, rpelvis = keypoints[6].copy(), keypoints[12].copy()
    sacrum_vector = lpelvis - rpelvis
    sacrum_center = keypoints[2, :-1] if keypoints[2, -1] > 0 else keypoints[[6, 12], :-1].mean(0)
    keypoints_ = keypoints[:, :-1] - sacrum_center
    
    rot = np.arctan2(sacrum_vector[2], sacrum_vector[0])

    x, y, z = np.split(keypoints_, 3, axis=1)
    z_ = z * np.cos(-rot) + x * np.sin(-rot)
    x_ = -z * np.sin(-rot) + x * np.cos(-rot)
    
    keypoints_[:, 0] = x_[:, 0]
    keypoints_[:, 2] = z_[:, 0]
    keypoints_ = keypoints_ + sacrum_center
    keypoints[:, :-1] = keypoints_
    
    return keypoints


def plot_3D_motion(keypoints, ax):
    from visualization_utils import get_parts, draw_ground, set_range
    
    keypoints = rotate_to_face_front(keypoints)
    center = keypoints[2, :-1] if keypoints[2, -1] > 0 else keypoints[[6, 12], :-1].mean(0)
    
    ax.set_axis_off()
    ax.view_init(azim = -90, elev = -50)

    draw_ground(ax)
    set_range(ax, offset=center.tolist())
    
    x, y, z, conf = np.split(keypoints, 4, axis=1)
    parts = get_parts()
    for part, value in parts.items():
        part_x = [x[idx][0] for idx in value[:-1] if conf[idx] > 0]
        part_y = [y[idx][0] for idx in value[:-1] if conf[idx] > 0]
        part_z = [z[idx][0] for idx in value[:-1] if conf[idx] > 0]

        ax.plot3D(part_x, part_y, part_z, value[-1])


def plot_signals(signal, idx, color, label, xrange, yrange, ax):
    
    idx_ = np.arange(idx)
    xlim = [idx - int(xrange/2), idx + int(xrange/2)]
    ylim = [1-int(yrange/2), 1+int(yrange/2)]
    ax.plot(idx_, signal[:idx], color=color, label=label)
    ax.scatter(idx, signal[idx])
    ax.legend()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def plot_and_save_analysis_fig(params, front_synced_imu_accel, front_kp_accel, 
                               end_synced_imu_accel, end_kp_accel):
    
    output_fldr = 'output/sync_analysis/'
    output_fldr= os.path.join(output_fldr, 'Set%02d'%params['id'], params['date'])
    
    if not os.path.exists(output_fldr):
        os.makedirs(output_fldr)
    
    fig = plt.figure(figsize=(10,5))
    front_ax, end_ax = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
    plt.suptitle("Syncing result (Left : front hopping, Right : End hopping)", 
                 y=0.95, fontsize=14)

    plot_sliding_with_index(0, front_synced_imu_accel, front_kp_accel, 
                            check_range=1, ax=front_ax)
    plot_sliding_with_index(0, end_synced_imu_accel, end_kp_accel, 
                            check_range=1, ax=end_ax)
    filename = 'exp%02d'%params['exp']
    plt.savefig(os.path.join(output_fldr, filename))
    plt.close()


def plot_sliding_with_index(index, imu_accel, kp_accel, check_range=200, ax=None):
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    start_idx, end_idx = index, index - check_range
    ax.plot(imu_accel[start_idx:end_idx], color='tab:red', label='IMU')
    ax.plot(kp_accel, color='tab:blue', label='KP')
    ax.legend()
    
    pass


def plot_convolutional_syncing_result(rmse_result, imu_accel, kp_accel, check_range):
    
    output_file = 'output/tmp.png'
    images = []

    pad = 30
    min_idx = np.where(rmse_result == rmse_result.min())[0][0]
    rmse_result_ = np.zeros(rmse_result.shape[0] + pad)
    rmse_result_[:min_idx] = rmse_result[:min_idx]
    rmse_result_[-(rmse_result.shape[0] - min_idx):] = rmse_result[min_idx:]
    rmse_result_[min_idx:-(rmse_result.shape[0] - min_idx)] = rmse_result.min()
    shift_idx = np.zeros(rmse_result.shape[0] + pad)
    shift_idx[:min_idx] = np.arange(min_idx)
    shift_idx[-(rmse_result.shape[0] - min_idx):] = np.arange(min_idx, rmse_result.shape[0])
    shift_idx[min_idx:-(rmse_result.shape[0] - min_idx)] = min_idx
    shift_idx = shift_idx.astype('int')

    with tqdm(total=rmse_result_.shape[0], leave=True) as prog_bar:
        for idx in shift_idx:
            # Generate subplots, 1) Accel data 2) RMSE of syncing
            fig = plt.figure(figsize=(10,5))
            sync_ax, rmse_ax = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
            plt.suptitle("IMU-Keypoints data syncing", y=0.95, fontsize=14)
            
            # Draw both subplots
            plot_sliding_with_index(idx, imu_accel, kp_accel, check_range=check_range, ax=sync_ax)
            rmse_ax.plot(rmse_result, color='tab:blue', label='RMSE')
            rmse_ax.scatter(idx, rmse_result[idx], color='tab:red')
            yrange = rmse_result.max() - rmse_result.min()
            ylim = [rmse_result.min() - 0.1 * yrange, rmse_result.max() + 0.1 * yrange]
            rmse_ax.legend()
            rmse_ax.set_ylim(ylim)
            
            # Save plots and load the figures by cv2
            plt.savefig(output_file)
            plt.close()
            im = cv2.imread(output_file)
            images += [im]
            prog_bar.update(1)
            prog_bar.refresh()

    generate_video(images)


def generate_video(images):
    output_file = 'output/animation.avi'
    height, width, _ = images[0].shape
    video = cv2.VideoWriter(output_file, 0, 15, (width, height))

    with tqdm(total=len(images), leave=True) as prog_bar:
        for image in images:
            video.write(image)
            prog_bar.update(1)
            prog_bar.refresh()
    
    cv2.destroyAllWindows()
    video.release()


def get_idx_from_params(params):
    subject_id = params['id']
    date = params['date']
    exp = params['exp']

    num_exps = [10, 8, 12, 14]
    date_list = ['190503', '190510', '190517', '190607']

    previous_exps = num_exps[:date_list.index(date)]
    idx_ = np.array(previous_exps).sum() + exp - 1
    idx_ += (subject_id - 1) * np.array(num_exps).sum()
    idx_ = idx_ - 1 if idx_ > 40 else idx_
    idx_ = idx_ - 1 if idx_ > 72 else idx_

    return idx_


def check_syncing_result_with_3D_model(params):
    import pickle
    from sync_data_utils import calculate_accel_from_keypoints, interpolate_kp_accel

    path_synced_data = params['save_dir']
    output_file = 'output/tmp.png'

    idx = get_idx_from_params(params)
    # Load synced data
    with open(os.path.join(path_synced_data, "keypoints.pkl"), "rb") as file:
        keypoints = pickle.load(file)[idx]
        keypoints = keypoints.cpu().detach().double().numpy()
    with open(os.path.join(path_synced_data, "accel.pkl"), "rb") as file:
        accel = pickle.load(file)[idx][:, -1, 1]
        accel = accel.cpu().detach().double().numpy()
    
    kp_accel_, _ = calculate_accel_from_keypoints(keypoints)
    kp_accel = pd.DataFrame(kp_accel_)[0]
    kp_accel = np.array(interpolate_kp_accel(kp_accel))

    images = []
    keypoints = keypoints[7800:7900]
    with tqdm(total=keypoints.shape[0], leave=True) as prog_bar:
        for frame in range(keypoints.shape[0]):
            fig = plt.figure(figsize=(10, 5))
            
            # ax1 : 3D motion video, ax2 : KP accel, ax3 : IMU accel
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 4)

            plot_3D_motion(keypoints[frame], ax1)
            plot_signals(kp_accel, 5 * frame, 'tab:blue', 'keypoints', 500, 8, ax2)
            plot_signals(accel, 5 * frame, 'tab:red', 'IMU', 500, 8, ax3)

            plt.savefig(output_file)
            plt.close()
            im = cv2.imread(output_file)
            images += [im]
            prog_bar.update(1)
            prog_bar.refresh()

    generate_video(images)
    

if __name__ == "__main__":
    import argparse
    from sync_data import define_configs

    parser = argparse.ArgumentParser()
    params = define_configs(parser)
    check_syncing_result_with_3D_model(params)