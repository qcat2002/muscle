import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

path = '/Users/zepeng/Project/muscle/processed_data/598/TS13/iso_0neutr_t05.mat'

data = loadmat(path)
ultra = data['Cut_Ultrasound']
# angle = data['Angle']
# print(data.keys())
#
# x = list(range(len(angle)))
# plt.plot(angle)
# plt.show()

import cv2
import os

h, w, _ = ultra.shape

# 定义视频编解码器并创建 VideoWriter 对象
cut = (598 - 363) // 2
left = cut
right = ultra.shape[1] - cut
os.makedirs(os.path.join(os.getcwd(), 'src', 'videos'), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID' for .avi, 'mp4v' for .mp4
out = cv2.VideoWriter('../src/videos/output598_crop.mp4', fourcc, 30, (right - left, h))  # 30 是帧率

# 逐帧写入
print(cut, left, right)
for slice_index in range(ultra.shape[2]):
    frame = ultra[:, left:right, slice_index]
    print(frame.shape)
    frame = np.uint8(255 * (frame - frame.min()) / (frame.max() - frame.min()))
    out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

out.release()
cv2.destroyAllWindows()
