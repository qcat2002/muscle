import numpy as np
from scipy.io import loadmat
import multiprocessing as mp
import tqdm
import os
import matplotlib.pyplot as plt

def mat_reader(f_path):
    angle_ = 'Angle'
    velocity_ = 'AngularVelocity'
    torque_ = 'Torque'
    images_ = 'Cut_Ultrasound'
    f_mat = loadmat(f_path)
    return f_mat[angle_], f_mat[velocity_], f_mat[torque_], f_mat[images_]

def get_angle(f_info):
    return f_info[0]

def get_velocity(f_info):
    return f_info[1]

def get_torque(f_info):
    return f_info[2]

def get_images(f_info):
    return f_info[3]

def get_slices(f_info):
    return f_info[3].shape[2]

def sample_rate_normalize(input_data, target_rate=601):
    input_data = input_data.flatten()
    f_original_len = len(input_data)
    # indices of sample points
    f_original_indices = np.linspace(0, f_original_len - 1, f_original_len)
    f_target_indices = np.linspace(0, f_original_len - 1, target_rate)
    f_target_indices = np.linspace(0, f_original_len - 1, target_rate)
    # extract information from original list to target list
    f_resampled_angles = np.interp(f_target_indices, f_original_indices, input_data)
    return f_resampled_angles, f_target_indices

if __name__ == "__main__":
    paths = ['/Users/zepeng/Project/muscle/processed_data/363/TS01_1/0deg_iso.mat',
             '/Users/zepeng/Project/muscle/processed_data/363/TS01_1/30deg_plant.mat',
             '/Users/zepeng/Project/muscle/processed_data/363/TS01_2/iso_0neutr_max.mat']
    mat_info = mat_reader(paths[0])
    print(mat_info[3].shape)
    angle_low_sample, indices = sample_rate_normalize(mat_info[0], target_rate=get_slices(mat_info))
    plt.figure(dpi=300)
    plt.title(f'Angle-Sample Rate Reduction-{len(angle_low_sample)} Samples')
    plt.plot(mat_info[0], label='Original Data')
    plt.plot(indices, angle_low_sample, label='Low Sample Rate Data')
    plt.legend()
    plt.savefig(os.path.join('src','readme_source','angle_low_sample_rate.png'))
    plt.show()

