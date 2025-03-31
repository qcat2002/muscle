from scipy.io import loadmat
import matplotlib.pyplot as plt


path1 = '/Users/zepeng/Project/muscle/processed_data/513/TS05/iso_0neutr_max.mat'
path2 = '/Users/zepeng/Project/muscle/processed_data/513/TS07/iso_0neutr_max.mat'
path3 = '/Users/zepeng/Project/muscle/processed_data/363/TS01_2/iso_0neutr_max.mat'
mat = loadmat(path2)
torque_small = mat['Torque']
# torque_small = torque_small * 150
print(torque_small.min(), torque_small.max())
mat2 = loadmat(path3)
torque_large = mat2['Torque']

plt.title('Range of Torque Values from different mat files')
plt.plot(torque_small)
plt.plot(torque_large)
plt.show()