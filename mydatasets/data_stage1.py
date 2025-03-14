import os
import random
import cv2
import kornia
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class SUIMDataset1(Dataset):
    # /root/autodl-tmp/data/ 4090
    # /root/autodl-tmp/project/datasets/maps/
    def __init__(self, data_path='/root/autodl-tmp/data/', transforms_=None , imsz=256):
        self.root_path_ = data_path
        self.data_path, self.label_path,self.com_path = self.random_sample()
        self.imsz = imsz
        self.size = len(self.data_path)
        self.trsform = transforms.Compose(transforms_)

    def random_sample(self):
        real_path_raw = self.root_path_ + 'trainA/'
        real_path_com = self.root_path_ + 'trainB/'
        real_path_ref = self.root_path_ + 'trainC/'

        img_names = os.listdir(real_path_raw)
        random.shuffle(img_names)
        img_names = img_names[:2250]
        real_data_path = [real_path_raw + p for p in img_names]
        real_com_path = [real_path_com + p for p in img_names]
        real_label_path = [real_path_ref + p for p in img_names]

        return real_data_path, real_label_path, real_com_path


    def __getitem__(self, item):
        img = cv2.imread(self.data_path[item])
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img_com = cv2.imread(self.com_path[item])
        # rgb_com = cv2.cvtColor(img_com, cv2.COLOR_BGR2RGB)

        img_label = cv2.imread(self.label_path[item])
        rgb_label = cv2.cvtColor(img_label, cv2.COLOR_BGR2RGB)

        rgb = torch.tensor(rgb.astype(float).transpose(2, 0, 1), dtype=torch.float) / 255.  # 3 h w
        # rgb_com = torch.tensor(rgb_com.astype(float).transpose(2, 0, 1), dtype=torch.float) / 255.  # 3 h w
        rgb_label = torch.tensor(rgb_label.astype(float).transpose(2, 0, 1), dtype=torch.float) / 255.  # 3 h w

        rgb = self.trsform(rgb)
        # rgb_com = self.trsform(rgb_com)
        rgb_label = self.trsform(rgb_label)

        return {'input': rgb, 'target': rgb_label}

    def __len__(self):
        return self.size
