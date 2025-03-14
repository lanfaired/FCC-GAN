import os

import kornia
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# 数据集的路径
dataset_path = 'D:\\Edge-download\\code\\classical_code\\mygan\\datasets\\maps\\trainC'

# 创建一个简单的转换，将PIL图像转换为张量
transform = transforms.Compose([
    transforms.ToTensor()
])

# 初始化累加器
mean = 0.0
sum_squared_diff = 0.0
num_pixels = 0

# 遍历数据集中的所有图像
for img_name in os.listdir(dataset_path):
    img_path = os.path.join(dataset_path, img_name)
    if os.path.isfile(img_path):
        # 使用PIL加载图像
        with Image.open(img_path) as img:
            img = img.convert('RGB')  # 确保图像是RGB格式
            img_tensor = transform(img)  # 转换为张量

            img_tensor = kornia.color.rgb_to_lab(img_tensor)
            img_tensor = img_tensor.view(3, -1)  # 重新形状为[3, H*W]
            mean += img_tensor.sum(1)  # 对所有像素求均值
            sum_squared_diff += (img_tensor ** 2).sum(1)  # 对所有像素求平方和
            num_pixels += img_tensor.size(1)  # 像素总数

# 计算最终的均值
mean /= num_pixels

# 计算最终的方差
variance = (sum_squared_diff / num_pixels) - (mean ** 2)

print(f'Mean: {mean}')
print(f'Variance: {variance}')