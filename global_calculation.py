from tqdm import tqdm
import numpy as np
from scipy.io import loadmat

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


def compute_global_mean_std(mat_paths):
    pixel_values = []
    for p in tqdm(mat_paths, desc="Computing mean/std"):
        us = loadmat(p)['Cut_Ultrasound']
        pixel_values.append(us.flatten())
    pixel_values = np.concatenate(pixel_values)
    return np.mean(pixel_values), np.std(pixel_values)

all_paths = path1 + path2 + path3
global_mean, global_std = compute_global_mean_std(all_paths)
print(f"Global mean: {global_mean:.4f}, std: {global_std:.4f}")