import os
import random
import cv2
import kornia
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class SUIMDataset(Dataset):
    # /root/autodl-tmp/data/ 4090
    # /root/autodl-tmp/project/datasets/maps/
    def __init__(self, data_path='/root/autodl-tmp/data/', imsz=256):
        self.root_path_ = data_path
        self.data_path, self.label_path,self.com_path = self.random_sample()
        self.imsz = imsz
        self.size = len(self.data_path)

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

        # random cropping
        h, w = rgb.shape[:2]
        th = self.imsz
        i = random.randint(0, h - th)
        j = random.randint(0, w - th)
        rgb = rgb[i:i + th, j:j + th]
        # rgb_com = rgb_com[i:i + th, j:j + th]
        rgb_label = rgb_label[i:i + th, j:j + th]

        # random rotation
        times = random.randint(0, 3)
        rgb = np.rot90(rgb, k=times)
        # rgb_com = np.rot90(rgb_com, k=times)
        rgb_label = np.rot90(rgb_label, k=times)

        if np.random.random() < 0.25:
            # 图像沿着1（y轴）翻转
            rgb = cv2.flip(rgb, 1)
            # rgb_com = cv2.flip(rgb_com, 1)
            rgb_label = cv2.flip(rgb_label, 1)

        rgb = torch.tensor(rgb.astype(float).transpose(2, 0, 1), dtype=torch.float) / 255.  # 3 h w
        # rgb_com = torch.tensor(rgb_com.astype(float).transpose(2, 0, 1), dtype=torch.float) / 255.  # 3 h w
        rgb_label = torch.tensor(rgb_label.astype(float).transpose(2, 0, 1), dtype=torch.float) / 255.  # 3 h w

        # lab = kornia.color.rgb_to_lab(rgb)  # l: 0~100; a,b: -127~127
        # l = lab[:1, :, :] / 100.  # 0~1
        # ab = lab[1:, :, :] / 127.  # -1~1
        # lab = torch.cat([l, ab], dim=0)

        # return {'input': rgb , 'target': rgb_label , 'lab':lab }
        return {'input': rgb, 'target': rgb_label}

    def __len__(self):
        return self.size

class GetValImage(Dataset):
    def __init__(self, root='/root/autodl-tmp/data/', transforms_=None):
        self.transform = transforms.Compose(transforms_)
        real_path_raw = root + 'testA/'
        real_path_com = root + 'testB/'
        real_path_ref = root + 'testC/'

        imgs = os.listdir(real_path_raw)
        self.len = len(imgs)
        labels = os.listdir(real_path_ref)

        self.real_data_path = [real_path_raw + p for p in imgs]
        self.real_label_path = [real_path_ref + p for p in labels]
        self.real_com_path = [real_path_com + p for p in labels]


    def __getitem__(self, item):
        img = cv2.imread(self.real_data_path[item])
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_com = cv2.imread(self.real_com_path[item])
        rgb_com = cv2.cvtColor(img_com, cv2.COLOR_BGR2RGB)

        img_label = cv2.imread(self.real_label_path[item])
        rgb_label = cv2.cvtColor(img_label, cv2.COLOR_BGR2RGB)

        rgb = torch.tensor(rgb.astype(float).transpose(2, 0, 1), dtype=torch.float) / 255.  # 3 h w
        rgb_com = torch.tensor(rgb_com.astype(float).transpose(2, 0, 1), dtype=torch.float) / 255.  # 3 h w
        rgb_label = torch.tensor(rgb_label.astype(float).transpose(2, 0, 1), dtype=torch.float) / 255.  # 3 h w

        rgb = self.transform(rgb)
        rgb_com = self.transform(rgb_com)
        rgb_label = self.transform(rgb_label)

        return {'image':rgb,'label': rgb_label,'com':rgb_com}

    def __len__(self):
        return self.len