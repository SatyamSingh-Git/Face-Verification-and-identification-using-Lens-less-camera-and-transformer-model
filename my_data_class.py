import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image


def npy_loader_blocks(path):
    sample = torch.from_numpy(np.load(path)).float()
    x1 = sample[0:3, :, :]
    x2 = sample[3:6, :, :]
    x3 = sample[6:9, :, :]
    x4 = sample[9:12, :, :]
    x5 = sample[12:15, :, :]
    return x1, x2, x3, x4, x5


class Lensless_DCT_offline(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.dataset = torchvision.datasets.DatasetFolder(root=data_path, loader=npy_loader_blocks, extensions=tuple(['.npy']))

    def __getitem__(self, idx):
        npy, label = self.dataset[idx]
        return npy, label

    def __len__(self):
        return len(self.dataset)


def noise_zero_out_ymdct(positions, ymdct):
    new_image_array = np.zeros_like(ymdct)
    new_image_array[:] = ymdct[:]
    for pos in positions:
        new_image_array[pos[0], pos[1], pos[2]*3:pos[2]*3+2] = 0
    return new_image_array


def npy_loader(path):
    sample = torch.from_numpy(np.load(path)).float()
    return sample

class Lensless_DCT_offline_noise(torch.utils.data.Dataset):
    def __init__(self, data_path, positions):
        self.dataset = torchvision.datasets.DatasetFolder(root=data_path, loader=npy_loader, extensions=tuple(['.npy']))
        self.positions = positions

    def __getitem__(self, idx):
        npy, label = self.dataset[idx]
        npy = noise_zero_out_ymdct(self.positions, np.transpose(npy, (1, 2, 0)))
        npy = np.transpose(npy, (2, 0, 1))
        x1 = npy[0:3, :, :]
        x2 = npy[3:6, :, :]
        x3 = npy[6:9, :, :]
        x4 = npy[9:12, :, :]
        x5 = npy[12:15, :, :]
        return (x1, x2, x3, x4, x5), label

    def __len__(self):
        return len(self.dataset)

