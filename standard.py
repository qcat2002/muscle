import numpy as np
import scipy.io
from scipy.special import gammainc
from scipy.optimize import minimize
import os

# ---------- Step 1: Helper to load ultrasound images from .mat files ----------
def load_ultrasound_images(mat_paths, key='Cut_Ultrasound'):
    """Load all ultrasound images from multiple .mat files into one array"""
    all_images = []
    for path in mat_paths:
        mat = scipy.io.loadmat(path)
        images = mat[key]  # [N, H, W]
        if images.ndim == 3:
            all_images.append(images)
        else:
            raise ValueError(f"Unexpected shape {images.shape} in {path}")
    return np.concatenate(all_images, axis=0)

# ---------- Step 2: Fit Generalized Gamma distribution parameters ----------
def generalized_gamma_neg_log_likelihood(params, data):
    """Negative log-likelihood of generalized gamma distribution"""
    m, s, sigma = params
    if m <= 0 or s <= 0 or sigma <= 0:
        return np.inf
    term1 = -m * s * np.log(data)
    term2 = (data / sigma) ** s
    nll = np.sum(term1 + term2)
    return nll

def fit_generalized_gamma(data, init=(2.0, 2.0, 30.0)):
    """Fit parameters using MLE"""
    data = data[data > 0]
    bounds = [(1e-2, None), (1e-2, None), (1e-2, None)]
    result = minimize(generalized_gamma_neg_log_likelihood, init, args=(data,), bounds=bounds)
    return result.x if result.success else init

# ---------- Step 3: Compute CDF and histogram matching ----------
def generalized_gamma_cdf(x, m, s, sigma):
    return gammainc(m, (x / sigma) ** s)

def build_lookup_table(src_image, m, s, sigma):
    src_hist, _ = np.histogram(src_image.flatten(), bins=256, range=(0, 255), density=True)
    src_cdf = np.cumsum(src_hist)
    src_cdf /= src_cdf[-1]
    ref_values = np.linspace(0, 255, 256)
    ref_cdf = generalized_gamma_cdf(ref_values, m, s, sigma)
    lookup = np.zeros(256)
    for i in range(256):
        target = src_cdf[i]
        lookup[i] = ref_values[np.argmin(np.abs(ref_cdf - target))]
    return lookup.astype(np.uint8)

def histogram_specification_image(image, lookup_table):
    return lookup_table[image]

# ---------- Step 4: Standardize B using parameters from A ----------
def standardize_dataset(path_B, params_A, save_dir=None):
    m, s, sigma = params_A
    for file_path in path_B:
        mat = scipy.io.loadmat(file_path)
        images = mat['Cut_Ultrasound']
        matched_images = np.stack([histogram_specification_image(img, build_lookup_table(img, m, s, sigma)) for img in images])

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            file_name = os.path.basename(file_path).replace('.mat', '_standardized.mat')
            save_path = os.path.join(save_dir, file_name)
            mat['Cut_Ultrasound'] = matched_images
            scipy.io.savemat(save_path, mat)
            print(f"Saved standardized file: {save_path}")
    return

# Pipeline entry
def run_pipeline(path1, path2):
    # Step 1: Load A's ultrasound data
    images_A = load_ultrasound_images(path1)
    flat_pixels = images_A.flatten().astype(np.float32)
    flat_pixels = flat_pixels[flat_pixels > 0] * 255  # scale to [0, 255] if needed

    # Step 2: Estimate GG parameters from A
    estimated_params = fit_generalized_gamma(flat_pixels)
    print(f"Estimated generalized gamma parameters from A: m={estimated_params[0]:.4f}, s={estimated_params[1]:.4f}, σ={estimated_params[2]:.4f}")

    # Step 3: Apply histogram specification to B
    save_directory = "./standardized_B"
    standardize_dataset(path2, estimated_params, save_dir=save_directory)

    return estimated_params

# This code is ready to be run inside your Python environment where the .mat files are accessible.
# Just call run_pipeline(path1, path2)

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