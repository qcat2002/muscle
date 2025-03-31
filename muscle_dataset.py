# muscle_dataset.py

import os
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import numpy as np
from glob import glob

class MuscleDataset(Dataset):
    def __init__(self, mat_dir, transform=None, target='Torque'):
        """
        :param mat_dir: directory with individual .mat files
        :param transform: optional transform to apply to each image
        :param target: which label to use: 'Angle', 'AngularVelocity', 'Torque'
        """
        self.files = sorted(glob(os.path.join(mat_dir, "*.mat")))
        self.transform = transform
        self.target_key = target
        self.samples = []

        # Build index: each sample is (file_path, frame_idx)
        for f in self.files:
            mat = loadmat(f)
            us = mat["Ultrasound"]  # shape: (H, W, T)
            T = us.shape[2]
            for t in range(T):
                self.samples.append((f, t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mat_path, frame_idx = self.samples[idx]
        mat = loadmat(mat_path)

        us = mat["Ultrasound"]  # (H, W, T)
        label = mat[self.target_key].flatten()  # (T,)

        img = us[:, :, frame_idx]  # (H, W)
        target = label[frame_idx]  # scalar

        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)  # normalize to [0, 1]
        img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(target, dtype=torch.float32)
