import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter
from preprocess import eliminate_passive_torque
from skimage import io, exposure

path1 = '/Users/zepeng/Project/muscle/processed_data/363/TS01_2/iso_0neutr_max.mat'
path2 = '/Users/zepeng/Project/muscle/processed_data/363/TS02/iso_0neutr_max.mat'

mat1 = loadmat(path1)
mat2 = loadmat(path2)

key = 'Cut_Ultrasound'
us1 = mat1[key]
us2 = mat2[key]
img1 = us1[:, :, 0]
img2 = us2[:, :, 0]
img2_ = exposure.match_histograms(img2, img1, channel_axis=None)


fig = plt.figure(figsize=(18, 6), dpi=300)
gs = gridspec.GridSpec(1, 3, width_ratios=[3, 1.5, 1.5], wspace=0.05)

# Histogram
ax0 = fig.add_subplot(gs[0])
ax0.set_title('Comparison of Intensity Histograms between\nvariant Ultrasound Images', fontweight='bold')
ax0.hist(img1.ravel(), bins=256, range=(0, 1), alpha=0.5, label='Ultrasound Image Type 1')
ax0.hist(img2_.ravel(), bins=256, range=(0, 1), alpha=0.5, label='Ultrasound Image Type 2 (Matched)')
ax0.set_xlabel('Intensity', fontsize=14, fontweight='bold')
ax0.set_ylabel('Frequency (Num of Pixels)', fontsize=14, fontweight='bold')
ax0.legend()

# Ultrasound Image 1
ax1 = fig.add_subplot(gs[1])
ax1.set_title('Ultrasound Image\nType 1', fontweight='bold')
ax1.imshow(img1, cmap='gray')
ax1.axis('off')

# Ultrasound Image 2
ax2 = fig.add_subplot(gs[2])
ax2.set_title('Matched Ultrasound Image\nType 2', fontweight='bold')
ax2.imshow(img2_, cmap='gray')
ax2.axis('off')
plt.savefig('src/readme_source/histgram_compare_matched.png', dpi=300, bbox_inches='tight')
plt.show()


# fig = plt.figure(figsize=(18, 6), dpi=300)
# gs = gridspec.GridSpec(1, 3, width_ratios=[3, 1.5, 1.5], wspace=0.05)
#
# # Histogram
# ax0 = fig.add_subplot(gs[0])
# ax0.set_title('Comparison of Intensity Histograms between\nvariant Ultrasound Images', fontweight='bold')
# ax0.hist(img1.ravel(), bins=256, range=(0, 1), alpha=0.5, label='Ultrasound Image Type 1')
# ax0.hist(img2.ravel(), bins=256, range=(0, 1), alpha=0.5, label='Ultrasound Image Type 2')
# ax0.set_xlabel('Intensity', fontsize=14, fontweight='bold')
# ax0.set_ylabel('Frequency (Num of Pixels)', fontsize=14, fontweight='bold')
# ax0.legend()
#
# # Ultrasound Image 1
# ax1 = fig.add_subplot(gs[1])
# ax1.set_title('Ultrasound Image\nType 1', fontweight='bold')
# ax1.imshow(img1, cmap='gray')
# ax1.axis('off')
#
# # Ultrasound Image 2
# ax2 = fig.add_subplot(gs[2])
# ax2.set_title('Ultrasound Image\nType 2', fontweight='bold')
# ax2.imshow(img2, cmap='gray')
# ax2.axis('off')
# plt.savefig('src/readme_source/histgram_compare_raw.png', dpi=300, bbox_inches='tight')
# plt.show()

# def denoise_signal(signal, window_length=21, polyorder=3):
#     """
#     :param signal: the signal need to denoise
#     :param window_length: length of window when denoising
#     :param polyorder: the degree of polynomial
#     :return: a denoised signal
#     """
#     return savgol_filter(signal, window_length, polyorder)
#
# paths = ['/Users/zepeng/Project/muscle/processed_data/363/TS01_2/iso_0neutr_max.mat',
#          '/Users/zepeng/Project/muscle/processed_data/363/TS01_2/iso_0neutr_t01.mat',
#          '/Users/zepeng/Project/muscle/processed_data/363/TS01_2/iso_0neutr_t02.mat',]
# fig, ax = plt.subplots(1, 2, dpi=300, figsize=(16, 6))
# max_, min_ = 0, 0
# for ii, path in enumerate(paths):
#     torque = loadmat(path)['Torque'].flatten()
#     denoised_torque = denoise_signal(torque, window_length=51, polyorder=3)
#     final_torque = eliminate_passive_torque(denoised_torque)
#     t_max = np.max(final_torque)
#     t_min = np.min(torque)
#     if t_max > max_:
#         max_ = t_max
#     if t_min < min_:
#         min_ = t_min
#     ax[0].plot(denoised_torque, label=f'Raw Torque (Denoised) {ii+1}')
#     ax[1].plot(final_torque, label=f"Corrected Torque {ii+1}")
# ax[0].axhline(0, linestyle='--', color='k')
# ax[1].axhline(0, linestyle='--', color='k')
# ax[0].set_ylim(min_, max_)
# ax[1].set_ylim(min_, max_)
# ax[0].set_xlabel('Sample Indices (Time)')
# ax[0].set_ylabel('Torque')
# ax[1].set_xlabel('Sample Indices (Time)')
# ax[1].set_ylabel('Torque')
# ax[0].legend()
# ax[1].legend()
# ax[0].set_title('Denoised Torque Tendency', fontsize=20, fontweight='bold')
# ax[1].set_title('Corrected Torque Tendency', fontsize=20, fontweight='bold')
# plt.tight_layout()
# plt.show()





# path1 = '/Users/zepeng/Project/muscle/processed_data/513/TS05/iso_0neutr_max.mat'
# path2 = '/Users/zepeng/Project/muscle/processed_data/513/TS07/iso_0neutr_max.mat'
# path3 = '/Users/zepeng/Project/muscle/processed_data/363/TS01_2/iso_0neutr_max.mat'
# path4 = '/Users/zepeng/Project/muscle/processed_data/363/TS01_2/iso_20pflx_t01.mat'
# path5 = '/Users/zepeng/Project/muscle/processed_data/363/TS01_2/iso_30pflx_t02.mat'
# mat = loadmat(path2)
# # torque_small = mat['Torque']
# # torque_small = torque_small * 150
# # print(torque_small.min(), torque_small.max())
# mat2 = loadmat(path3)
# torque_large = mat2['Torque']
# mat3 = loadmat(path4)
# torque_large2 = mat3['Torque']
# mat4 = loadmat(path5)
# torque_large3 = mat4['Torque']
#
# # plt.title('Range of Torque Values from different mat files')
# # plt.title('Example of Small Torque Values')
# plt.figure(dpi=300)
# plt.title('Set a threshold to classify Active Contraction State')
# # plt.plot(torque_small)
# plt.axhline(y=5, color='red', linestyle='--', linewidth=1)
# plt.xlabel('Sample Indices')
# plt.ylabel('Torque Values')
# plt.plot(torque_large, label='Torque1')
# plt.plot(torque_large2, label='Torque2')
# plt.plot(torque_large3, label='Torque3')
# plt.legend()
# plt.show()