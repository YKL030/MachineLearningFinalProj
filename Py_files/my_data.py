import numpy as np
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import torch
import torchvision
import pandas
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid


# step1: 定义MyDataset类， 继承Dataset, 重写抽象方法：__len()__, __getitem()__
class MyDataset(Dataset):

    def __init__(self, root_dir, root_dir1, names_file, transform=None):
        self.root_dir = root_dir
        self.root_dir1 = root_dir1
        self.names_file = self.root_dir + names_file
        self.transform = transform
        self.size = 0
        self.names_list = []

        if not os.path.isfile(self.names_file):
            print(self.names_file + 'does not exist!')
        file = open(self.names_file)
        for f in file:
            self.names_list.append(f)
            self.size += 1

        del self.names_list[0]
        self.size -=1
        print(self.size)


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        #print(idx)
        candidate_path = self.root_dir + self.root_dir1 + self.names_list[idx].split(',')[0] + ('.npz')
        if not os.path.isfile(candidate_path):
            print(candidate_path + 'does not exist!')
            return None
        data = np.load(candidate_path)   # use skitimage
        voxel = data['voxel'].astype(np.float32)
        seg = data['seg'].astype(np.float32)
        voxel = voxel*seg
        voxel = voxel[30:62,30:62,30:62]
        #seg = data['seg'][30:61,30:61,30:61].astype(np.float32)
        label = float(self.names_list[idx].split(',')[1])

        Min = 0
        Max = 255
        voxel = (voxel - Min) / (Max - Min)

        return voxel,label

