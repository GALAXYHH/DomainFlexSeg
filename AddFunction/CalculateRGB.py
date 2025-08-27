import os
import numpy as np
from PIL import Image

def calculate_mean_and_std(image_folder):
    # 初始化存储像素值的列表
    pixel_values = []

    # 遍历文件夹中的所有子文件夹
    for subdir in os.listdir(image_folder):
        subdir_path = os.path.join(image_folder, subdir)
        if os.path.isdir(subdir_path):  # 确保是子文件夹
            # 遍历子文件夹中的所有子文件夹
            for inner_subdir in os.listdir(subdir_path):
                inner_subdir_path = os.path.join(subdir_path, inner_subdir)
                if os.path.isdir(inner_subdir_path):  # 确保是子文件夹
                    # 遍历子文件夹中的所有文件
                    for filename in os.listdir(inner_subdir_path):
                        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')) and 'mask' not in filename.lower():
                            img_path = os.path.join(inner_subdir_path, filename)
                            img = Image.open(img_path).convert('RGB')  # 转换为 RGB 格式
                            img_array = np.array(img)  # 转换为 numpy 数组
                            pixel_values.append(img_array)

    # 检查是否有有效图片
    if not pixel_values:
        raise ValueError("没有找到有效的图片文件。")

    # 将所有图像的像素值合并为一个大数组
    all_pixels = np.concatenate([img.reshape(-1, 3) for img in pixel_values], axis=0)

    # 计算均值和标准差
    mean = np.mean(all_pixels, axis=0) / 255.0  # 除以 255 归一化到 [0, 1]
    std = np.std(all_pixels, axis=0) / 255.0  # 除以 255 归一化

    return mean, std

# 使用示例
image_folder  = 'C:\\Users\\19551\\Desktop\\Domain\\MDF-Net-master\\data'  # 替换为你的文件夹路径
mean, std = calculate_mean_and_std(image_folder)
print("Mean:", mean)
print("Standard Deviation:", std)