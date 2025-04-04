from scipy.io import loadmat
import matplotlib.pyplot as plt


path1 = '/Users/zepeng/Project/muscle/processed_data/513/TS05/iso_0neutr_max.mat'
path2 = '/Users/zepeng/Project/muscle/processed_data/513/TS07/iso_0neutr_max.mat'
path3 = '/Users/zepeng/Project/muscle/processed_data/363/TS01_2/iso_0neutr_max.mat'
path4 = '/Users/zepeng/Project/muscle/processed_data/363/TS01_2/iso_20pflx_t01.mat'
path5 = '/Users/zepeng/Project/muscle/processed_data/363/TS01_2/iso_30pflx_t02.mat'
mat = loadmat(path2)
# torque_small = mat['Torque']
# torque_small = torque_small * 150
# print(torque_small.min(), torque_small.max())
mat2 = loadmat(path3)
torque_large = mat2['Torque']
mat3 = loadmat(path4)
torque_large2 = mat3['Torque']
mat4 = loadmat(path5)
torque_large3 = mat4['Torque']

# plt.title('Range of Torque Values from different mat files')
# plt.title('Example of Small Torque Values')
plt.figure(dpi=300)
plt.title('Set a threshold to classify Active Contraction State')
# plt.plot(torque_small)
plt.axhline(y=5, color='red', linestyle='--', linewidth=1)
plt.xlabel('Sample Indices')
plt.ylabel('Torque Values')
plt.plot(torque_large, label='Torque1')
plt.plot(torque_large2, label='Torque2')
plt.plot(torque_large3, label='Torque3')
plt.legend()
plt.show()