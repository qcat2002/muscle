import numpy as np
from scipy.io import loadmat, savemat
from multiprocessing import Pool
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path
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
    """
    :param signal: the signal need to denoise
    :param window_length: length of window when denoising
    :param polyorder: the degree of polynomial
    :return: a denoised signal
    """
    return savgol_filter(signal, window_length, polyorder)

def get_angle(f_info):
    """
    :param f_info: some dictionary objects, which read from mat files
    :return: angle numpy array, had already flatten()
    """
    return denoise_signal(f_info[0].flatten(), window_length=151, polyorder=2)

def get_velocity(f_info):
    """
    :param f_info: some dictionary objects, which read from mat files
    :return: velocity numpy array, had already flatten()
    """
    return denoise_signal(f_info[1].flatten(), window_length=151, polyorder=2)

def get_torque(f_info):
    """
    :param f_info: some dictionary objects, which read from mat files
    :return: torque numpy array, had already flatten()
    """
    return denoise_signal(f_info[2].flatten(), window_length=51, polyorder=2)

def get_images(f_info):
    """
    :param f_info: some dictionary objects, which read from mat files
    :return: ultrasound image [: (height), : (width), : (slice channel)], had already flatten()
    """
    return f_info[3]

def get_slices(f_info):
    """
    :param f_info: some dictionary objects, which read from mat files
    :return: the number of slices in mat file. Usually it is 601 slices.
    """
    return f_info[3].shape[2]


def sample_rate_normalize(input_data, target_rate=601, kind='cubic'):
    """
    :param input_data: high sample rate information, such as angles, velocities, torques
    :param target_rate: the sample rate, which is same as the number of slices
    :param kind: interpolation method
    :return: a low sample rate data (signal)
    """
    input_data = input_data.flatten()
    f_original_len = len(input_data)
    f_original_indices = np.linspace(0, f_original_len - 1, f_original_len)
    f_target_indices = np.linspace(0, f_original_len - 1, target_rate)

    # 使用更高阶插值方法
    interpolator = interp1d(f_original_indices, input_data, kind=kind, fill_value="extrapolate")
    f_resampled = interpolator(f_target_indices)

    return f_resampled, f_target_indices

def name_extractor(f_path):
    """
    :param f_path: path of mat file
    :return: specific name for this mat file (folder name + experiment_attributes)
    """
    path = Path(f_path)
    return f"{path.parent.name}-{path.stem}"

def display_as_video(f_frames, f_datas, f_data_types, f_data_units, video_name=''):
    """
    :param f_frames: ultrasound images
    :param f_datas: list of information we require to show
    :param f_data_types: list of names (of information) we require to show
    :param video_name: name of the video
    :return: None
    """
    original_height, original_width = f_frames[:, :, 0].shape
    text_area_width = 370
    total_width = original_width + text_area_width

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join('src', 'ultrasound_videos')
    os.makedirs(video_path, exist_ok=True)
    file_path = os.path.join(video_path, f'{video_name}.mp4')
    out = cv.VideoWriter(file_path, fourcc, 25, (total_width, original_height))
    if not out.isOpened():
        raise IOError(f"[ERROR] VideoWriter failed to open! Check codec or FPS. Path: {file_path}")

    for fi, img_index in tqdm(enumerate(range(f_frames.shape[2])), desc='Adding Frames ...'):
        i_frame = f_frames[:, :, img_index]
        if i_frame.dtype != np.uint8:
            if i_frame.max() <= 1.0:
                i_frame = (i_frame * 255).astype(np.uint8)
            else:
                i_frame = cv.normalize(i_frame, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        i_frame = cv.cvtColor(i_frame, cv.COLOR_GRAY2BGR)

        canvas = np.zeros((original_height, total_width, 3), dtype=np.uint8)
        canvas[:, :original_width, :] = i_frame

        x_origin = original_width + 10
        y_origin = 60
        for (data, data_name, unit) in zip(f_datas, f_data_types, f_data_units):
            text = f"| {data_name}: {data[fi]:.2f} ({unit})"
            cv.putText(canvas, text, (x_origin, y_origin), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            y_origin += 40

        out.write(canvas)

    out.release()

def normalize_ultrasound(us):
    global_mean = 0.3341
    global_std = 0.2158
    us = us.astype(np.float32)
    return (us - global_mean) / (global_std + 1e-5)


def process_one_file(f_path):
    try:
        angle, vel, torque, us = mat_reader(f_path)
        rate = us.shape[2]
        my_name = name_extractor(f_path)
        # normalize the image
        us = normalize_ultrasound(us)
        angle, _ = sample_rate_normalize(denoise_signal(angle.flatten(), 151, 2), target_rate=rate)
        vel, _ = sample_rate_normalize(denoise_signal(vel.flatten(), 151, 2), target_rate=rate)
        torque, _ = sample_rate_normalize(denoise_signal(torque.flatten(), 51, 2), target_rate=rate)
        display_as_video(us,
                         f_datas=[angle, vel, torque],
                         f_data_types=['Angle', 'Velocity', 'Torque'],
                         f_data_units=['deg', 'deg/s', 'N*m'],
                         video_name=my_name)
        return {
            "Angle": angle,
            "AngularVelocity": vel,
            "Torque": torque,
            "Ultrasound": us,
            "Name": name_extractor(f_path)
        }
    except Exception as e:
        print(f"[ERROR] Failed to process {f_path}: {e}")
        return None

if __name__ == "__main__":
    path1 = [
        '/Users/zepeng/Project/muscle/processed_data/363/TS01_2/iso_0neutr_max.mat',
        '/Users/zepeng/Project/muscle/processed_data/363/TS01_2/iso_0neutr_t01.mat',
        '/Users/zepeng/Project/muscle/processed_data/363/TS01_2/iso_0neutr_t02.mat',
        '/Users/zepeng/Project/muscle/processed_data/363/TS01_2/iso_10dflx_max.mat',
        '/Users/zepeng/Project/muscle/processed_data/363/TS01_2/iso_10dflx_t01.mat',
        '/Users/zepeng/Project/muscle/processed_data/363/TS01_2/iso_10dflx_t02.mat',
        '/Users/zepeng/Project/muscle/processed_data/363/TS01_2/iso_10pflx_max.mat',
        '/Users/zepeng/Project/muscle/processed_data/363/TS01_2/iso_10pflx_t02.mat',
        '/Users/zepeng/Project/muscle/processed_data/363/TS01_2/iso_20pflx_max.mat',
        '/Users/zepeng/Project/muscle/processed_data/363/TS01_2/iso_20pflx_t01.mat',
        '/Users/zepeng/Project/muscle/processed_data/363/TS01_2/iso_20pflx_t02.mat',
        '/Users/zepeng/Project/muscle/processed_data/363/TS01_2/iso_30pflx_max.mat',
        '/Users/zepeng/Project/muscle/processed_data/363/TS01_2/iso_30pflx_t01.mat',
        '/Users/zepeng/Project/muscle/processed_data/363/TS01_2/iso_30pflx_t02.mat',
    ]
    path2 = [
        "/Users/zepeng/Project/muscle/processed_data/363/TS02/iso_0neutr_max.mat",
        "/Users/zepeng/Project/muscle/processed_data/363/TS02/iso_0neutr_t01.mat",
        "/Users/zepeng/Project/muscle/processed_data/363/TS02/iso_0neutr_t02.mat",
        "/Users/zepeng/Project/muscle/processed_data/363/TS02/iso_10dflx_max.mat",
        "/Users/zepeng/Project/muscle/processed_data/363/TS02/iso_10dflx_t01.mat",
        "/Users/zepeng/Project/muscle/processed_data/363/TS02/iso_10dflx_t02.mat",
        "/Users/zepeng/Project/muscle/processed_data/363/TS02/iso_10pflx_max.mat",
        "/Users/zepeng/Project/muscle/processed_data/363/TS02/iso_10pflx_t01.mat",
        "/Users/zepeng/Project/muscle/processed_data/363/TS02/iso_10pflx_t02.mat",
        "/Users/zepeng/Project/muscle/processed_data/363/TS02/iso_20pflx_max.mat",
        "/Users/zepeng/Project/muscle/processed_data/363/TS02/iso_20pflx_t01.mat",
        "/Users/zepeng/Project/muscle/processed_data/363/TS02/iso_20pflx_t02.mat",
        "/Users/zepeng/Project/muscle/processed_data/363/TS02/iso_30pflx_max.mat",
        "/Users/zepeng/Project/muscle/processed_data/363/TS02/iso_30pflx_t01.mat",
        "/Users/zepeng/Project/muscle/processed_data/363/TS02/iso_30pflx_t02.mat"
    ]



    os.makedirs("src/processed_dataset", exist_ok=True)
    path_decision = path2
    with Pool(processes=os.cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_one_file, path_decision), total=len(path_decision), desc="Processing files"))

    for r in results:
        if r is None:
            continue
        save_path = os.path.join("src/processed_dataset", f"{r['Name']}.mat")
        savemat(save_path, {
            "Angle": r["Angle"],
            "AngularVelocity": r["AngularVelocity"],
            "Torque": r["Torque"],
            "Ultrasound": r["Ultrasound"],
            "Name": r["Name"]
        }, do_compression=True)
        print(f"[INFO] Saved: {save_path}")
