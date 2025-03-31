# from scipy.io import loadmat
# import matplotlib.pyplot as plt
#
# file = '/Users/zepeng/Project/muscle/processed_data/363/TS01_2/iso_10dflx_t01.mat'
# data = loadmat(file)
# print(data.keys())
# images = data['Cut_Ultrasound']
# angles = data['Angle']
# torque = data['Torque']
# velocity = data['AngularVelocity']
#
# length_data = torque.shape[0]
# num_slices = images.shape[2]
# print(length_data, num_slices)
# print(length_data // num_slices)
# print(length_data / num_slices)
