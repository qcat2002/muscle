import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat

def find_mat_files(f_width=363):
    """
    :param f_width: 默认 363 的图像宽度
    :return: 所有的文件绝对路径 [LIST]
    """
    f_base_path = os.path.join(os.getcwd(), 'processed_data')
    f_targets = os.path.join(f_base_path, f'{f_width}', '**', '*.mat')
    files = glob.glob(f_targets, recursive=True)
    return files

class MuscleDataset(Dataset):
    def __init__(self, c_batch_size=8, c_width=363):
        self.file_list = find_mat_files(c_width)
        self.c_batch_size = c_batch_size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, f_idx):
        f_file = self.file_list[f_idx]
        f_data = loadmat(f_file)
        f_ultrasound_img = torch.tensor(f_data['Cut_Ultrasound'], dtype=torch.float32)
        f_torque = torch.tensor(f_data['Torque'], dtype=torch.float32)
        return

if __name__ == '__main__':
    dataset = MuscleDataset()
    dataloader = DataLoader(dataset, batch_size=8, num_workers=4, shuffle=False)
    for x, y in dataloader:
        print(x.shape, y.shape)