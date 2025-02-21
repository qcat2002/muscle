from scipy.io import loadmat

path = '/Users/zepeng/Project/muscle/processed_data/513/TS09/iso_10dflx_max.mat'

data = loadmat(path)
ultra = data['Cut_Ultrasound']
print(ultra.shape)