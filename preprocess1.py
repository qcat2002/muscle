import numpy as np
from scipy.io import loadmat
import multiprocessing as mp
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import cv2 as cv
from plots import plot_denoised_data_with_low_sample_rate

def mat_reader(f_path):
    angle_ = 'Angle'
    velocity_ = 'AngularVelocity'
    torque_ = 'Torque'
    images_ = 'Cut_Ultrasound'
    f_mat = loadmat(f_path)
    return f_mat[angle_], f_mat[velocity_], f_mat[torque_], f_mat[images_]

def denoise_signal(signal, window_length=21, polyorder=3):
    return savgol_filter(signal, window_length, polyorder)

def get_angle(f_info):
    return denoise_signal(f_info[0].flatten(), window_length=91, polyorder=3)

def get_velocity(f_info):
    return denoise_signal(f_info[1].flatten(), window_length=111, polyorder=3)

def get_torque(f_info):
    return f_info[2]

def get_images(f_info):
    return f_info[3]

def get_slices(f_info):
    return f_info[3].shape[2]


def sample_rate_normalize(input_data, target_rate=601, kind='cubic'):
    input_data = input_data.flatten()
    f_original_len = len(input_data)
    f_original_indices = np.linspace(0, f_original_len - 1, f_original_len)
    f_target_indices = np.linspace(0, f_original_len - 1, target_rate)

    # 使用更高阶插值方法
    interpolator = interp1d(f_original_indices, input_data, kind=kind, fill_value="extrapolate")
    f_resampled = interpolator(f_target_indices)

    return f_resampled, f_target_indices

def display_as_video(f_frames, f_datas, f_data_types, video_name=''):
    height, width = f_frames[:, :, 0].shape
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(video_name, fourcc, 10, (width, height))

    for fi, img_index in tqdm(enumerate(range(f_frames.shape[2])), desc='Adding Frames ...'):
        i_frame = f_frames[:, :, img_index]
        if i_frame.dtype != np.uint8:
            if i_frame.max() <= 1.0:
                i_frame = (i_frame * 255).astype(np.uint8)
            else:
                i_frame = cv.normalize(i_frame, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        i_frame = cv.cvtColor(i_frame, cv.COLOR_GRAY2BGR)
        y_origin = 60
        for data, data_name in zip(f_datas, f_data_types):
            new_text = f"|{data_name}({data[fi]:.2f})"
            cv.putText(i_frame, new_text, (20, y_origin), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            y_origin += 40
        out.write(i_frame)
    out.release()

    return

if __name__ == "__main__":
    paths = ['/Users/zepeng/Project/muscle/processed_data/363/TS01_1/0deg_iso.mat',
             '/Users/zepeng/Project/muscle/processed_data/363/TS01_1/30deg_plant.mat',
             '/Users/zepeng/Project/muscle/processed_data/363/TS01_2/iso_0neutr_max.mat',
             '/Users/zepeng/Project/muscle/processed_data/363/TS01_2/iso_10dflx_max.mat',
             '/Users/zepeng/Project/muscle/processed_data/363/TS01_2/iso_10pflx_max.mat',]
    mat_info = mat_reader(paths[3])
    name = 'up_test1.mp4'
    rate = get_slices(mat_info)
    angle_low_sample, angle_indices = sample_rate_normalize(get_angle(mat_info), target_rate=rate)
    plot_denoised_data_with_low_sample_rate(get_angle(mat_info), angle_low_sample, angle_indices, data_name='Angle')
    velocity_low_sample, velocity_indices = sample_rate_normalize(get_velocity(mat_info), target_rate=rate)
    plot_denoised_data_with_low_sample_rate(get_velocity(mat_info), velocity_low_sample, velocity_indices, data_name='Velocity')

    display_as_video(get_images(mat_info), [angle_low_sample, velocity_low_sample], f_data_types=['Angle', 'Velocity'], video_name=name)








    """
    Old plot codes
    """
    # angle_low_sample, indices = sample_rate_normalize(mat_info[0], target_rate=get_slices(mat_info))
    # plt.figure(dpi=300)
    # plt.title(f'Angle-Sample Rate Reduction-{len(angle_low_sample)} Samples')
    # plt.plot(mat_info[0], label='Original Data')
    # plt.plot(indices, angle_low_sample, label='Low Sample Rate Data')
    # plt.legend()
    # plt.savefig(os.path.join('src','readme_source','angle_low_sample_rate.png'))
    # plt.show()

