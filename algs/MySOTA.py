# coding=utf-8

import os
import random
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
from evaluation import get_DC


def mixup_images(tensor, alpha=1):
    #assert tensor.shape[0] == 8, "输入的 tensor 第一维度应该为 8"
    lent = round(tensor.shape[0]/2)
    lams = np.random.beta(alpha, alpha, size=lent)  # 生成4个随机的 lam 值
    mixed_images = torch.zeros(lent, 3, 320, 320)
    for i in range(lent):
        mixed_images[i] = lams[i] * tensor[i] + (1 - lams[i]) * tensor[i + lent]
    return mixed_images, lams

def mix_loss(criterion, output, x0_c, x1_c, x2_c, x3_c, x4_c, target, Up2, Up4, Up8, Up16, lams):
    mix_loss = 0
    for i in range(len(lams)):
        loss1 = criterion(output[i], target[i]) + criterion(x0_c[i], target[i]) + criterion(Up2(x1_c)[i], target[i]) + criterion(Up4(x2_c)[i], target[i]) + criterion(Up8(x3_c)[i], target[i]) + criterion(Up16(x4_c)[i], target[i])
        loss2 = criterion(output[i], target[i+len(lams)]) + criterion(x0_c[i], target[i+len(lams)]) + criterion(Up2(x1_c)[i], target[i+len(lams)]) + criterion(Up4(x2_c)[i], target[i+len(lams)]) + criterion(Up8(x3_c)[i], target[i+len(lams)]) + criterion(Up16(x4_c)[i], target[i+len(lams)])
        mix_loss += lams[i]*loss1 + (1-lams[i])*loss2
    return mix_loss

def mix_dice(output, target, lams):
    mix_dice = 0
    for i in range(len(lams)):
        dice1 = get_DC(output.ge(0.5).float()[i], target[i])
        dice2 = get_DC(output.ge(0.5).float()[i], target[i+len(lams)])
        mix_dice += lams[i] * dice1 + (1-lams[i]) * dice2
    return mix_dice

def cutmix_images(imagesin, targetsin, alpha=1.0):
    #assert imagesin.shape == (8, 3, 320, 320), "输入tensor的形状应为 [8, 3, 320, 320]"
    lent = round(imagesin.shape[0]/2)
    output_images = torch.empty((lent, 3, 320, 320))  # 创建空的输出tensor
    output_targets = torch.empty((lent, 320, 320))  # 创建空的输出tensor
    lams = np.random.beta(alpha, alpha, size=lent)  # 生成4个随机的 lam 值
    #𝛼=1.0和 𝛽=1.0：这种设置对应于均匀分布，生成的切割比例会比较均匀，适合一般情况下的数据增强。
    #𝛼=0.2和 β=0.2：这种设置会导致较小的切割区域，更加集中于从一张图像中提取大部分内容，适合需要保留更多原始内容的场景。
    #𝛼=2.0和 β=5.0：这种设置倾向于生成较大的切割区域，适合需要更多样本变换的情况。
    for i in range(lent):
        idx1, idx2 = i, i + lent
        h, w = imagesin.shape[2], imagesin.shape[3]
        cut_area = lams[i] * h * w
        cut_w = int(np.sqrt(cut_area))
        cut_h = int(np.sqrt(cut_area))
        cut_x = np.random.randint(0, w - cut_w)
        cut_y = np.random.randint(0, h - cut_h)
        new_image = imagesin[idx2].clone()
        new_target = targetsin[idx2].clone()
        new_image[:, cut_y:cut_y + cut_h, cut_x:cut_x + cut_w] = imagesin[idx1][:, cut_y:cut_y + cut_h, cut_x:cut_x + cut_w]
        new_target[cut_y:cut_y + cut_h, cut_x:cut_x + cut_w] = targetsin[idx1][cut_y:cut_y + cut_h, cut_x:cut_x + cut_w]
        output_images[i] = new_image
        output_targets[i] = new_target
    return output_images, output_targets, lams

def cutmix_images_large(imagesin, targetsin):
    lent = round(imagesin.shape[0]/2)
    output_images = torch.empty((lent, 3, 320, 320))  # 创建空的输出tensor
    output_targets = torch.empty((lent, 320, 320))  # 创建空的输出tensor
    lams = np.random.beta(2, 5, size=lent)  # 生成4个随机的 lam 值
    #𝛼=1.0和 𝛽=1.0：这种设置对应于均匀分布，生成的切割比例会比较均匀，适合一般情况下的数据增强。
    #𝛼=0.2和 β=0.2：这种设置会导致较小的切割区域，更加集中于从一张图像中提取大部分内容，适合需要保留更多原始内容的场景。
    #𝛼=2.0和 β=5.0：这种设置倾向于生成较大的切割区域，适合需要更多样本变换的情况。
    for i in range(lent):
        idx1, idx2 = i, i + lent
        h, w = imagesin.shape[2], imagesin.shape[3]
        cut_area = lams[i] * h * w
        cut_w = int(np.sqrt(cut_area))
        cut_h = int(np.sqrt(cut_area))
        cut_x = np.random.randint(0, w - cut_w)
        cut_y = np.random.randint(0, h - cut_h)
        new_image = imagesin[idx2].clone()
        new_target = targetsin[idx2].clone()
        new_image[:, cut_y:cut_y + cut_h, cut_x:cut_x + cut_w] = imagesin[idx1][:, cut_y:cut_y + cut_h, cut_x:cut_x + cut_w]
        new_target[cut_y:cut_y + cut_h, cut_x:cut_x + cut_w] = targetsin[idx1][cut_y:cut_y + cut_h, cut_x:cut_x + cut_w]
        output_images[i] = new_image
        output_targets[i] = new_target
    return output_images, output_targets, lams

def cutmix_images_little(imagesin, targetsin):
    lent = round(imagesin.shape[0]/2)
    output_images = torch.empty((lent, 3, 320, 320))  # 创建空的输出tensor
    output_targets = torch.empty((lent, 320, 320))  # 创建空的输出tensor
    lams = np.random.beta(0.2, 0.2, size=lent)  # 生成4个随机的 lam 值
    #𝛼=1.0和 𝛽=1.0：这种设置对应于均匀分布，生成的切割比例会比较均匀，适合一般情况下的数据增强。
    #𝛼=0.2和 β=0.2：这种设置会导致较小的切割区域，更加集中于从一张图像中提取大部分内容，适合需要保留更多原始内容的场景。
    #𝛼=2.0和 β=5.0：这种设置倾向于生成较大的切割区域，适合需要更多样本变换的情况。
    for i in range(lent):
        idx1, idx2 = i, i + lent
        h, w = imagesin.shape[2], imagesin.shape[3]
        cut_area = lams[i] * h * w
        cut_w = int(np.sqrt(cut_area))
        cut_h = int(np.sqrt(cut_area))
        if w != cut_w:
            cut_x = np.random.randint(0, w - cut_w)
            cut_y = np.random.randint(0, h - cut_h)
        else:
            cut_x = 0
            cut_y = 0
        new_image = imagesin[idx2].clone()
        new_target = targetsin[idx2].clone()
        new_image[:, cut_y:cut_y + cut_h, cut_x:cut_x + cut_w] = imagesin[idx1][:, cut_y:cut_y + cut_h, cut_x:cut_x + cut_w]
        new_target[cut_y:cut_y + cut_h, cut_x:cut_x + cut_w] = targetsin[idx1][cut_y:cut_y + cut_h, cut_x:cut_x + cut_w]
        output_images[i] = new_image
        output_targets[i] = new_target
    return output_images, output_targets, lams


def cutmix_images_large_mix_rpair(imagesin, targetsin):
    lent = imagesin.shape[0]

    output_images = torch.empty((lent, 3, 320, 320))  # 创建空的输出 tensor
    output_targets = torch.empty((lent, 320, 320))  # 创建空的输出 tensor

    # 随机打乱索引以获取配对
    indices = np.random.permutation(lent)

    # 处理配对的图像
    for i in range(0, lent, 2):  # 每次处理两张图像
        idx1 = indices[i]
        idx2 = indices[i + 1]

        h, w = imagesin.shape[2], imagesin.shape[3]

        # 生成切割比例
        lam = np.random.beta(2, 5)
        cut_area = lam * h * w
        cut_w = int(np.sqrt(cut_area))
        cut_h = int(np.sqrt(cut_area))

        # 随机选择切割区域的起始位置
        cut_x = np.random.randint(0, w - cut_w)
        cut_y = np.random.randint(0, h - cut_h)

        # 克隆新的图像和目标
        new_image1 = imagesin[idx1].clone()
        new_image2 = imagesin[idx2].clone()
        new_target1 = targetsin[idx1].clone()
        new_target2 = targetsin[idx2].clone()

        # 替换切割区域
        new_image1[:, cut_y:cut_y + cut_h, cut_x:cut_x + cut_w] = imagesin[idx2][:, cut_y:cut_y + cut_h, cut_x:cut_x + cut_w]
        new_target1[cut_y:cut_y + cut_h, cut_x:cut_x + cut_w] = targetsin[idx2][cut_y:cut_y + cut_h, cut_x:cut_x + cut_w]
        new_image2[:, cut_y:cut_y + cut_h, cut_x:cut_x + cut_w] = imagesin[idx1][:, cut_y:cut_y + cut_h, cut_x:cut_x + cut_w]
        new_target2[cut_y:cut_y + cut_h, cut_x:cut_x + cut_w] = targetsin[idx1][cut_y:cut_y + cut_h, cut_x:cut_x + cut_w]

        # 存储新的图像和目标
        output_images[i] = new_image1
        output_images[i + 1] = new_image2
        output_targets[i] = new_target1
        output_targets[i + 1] = new_target2

    return output_images, output_targets

def cutmix_images_mix_rpair(imagesin, targetsin, alpha=1.0):
    lent = imagesin.shape[0]

    output_images = torch.empty((lent, 3, 320, 320))  # 创建空的输出 tensor
    output_targets = torch.empty((lent, 320, 320))  # 创建空的输出 tensor

    # 随机打乱索引以获取配对
    indices = np.random.permutation(lent)

    # 处理配对的图像
    for i in range(0, lent, 2):  # 每次处理两张图像
        idx1 = indices[i]
        idx2 = indices[i + 1]

        h, w = imagesin.shape[2], imagesin.shape[3]

        # 生成切割比例
        lam = np.random.beta(alpha, alpha)
        cut_area = lam * h * w
        cut_w = int(np.sqrt(cut_area))
        cut_h = int(np.sqrt(cut_area))

        # 随机选择切割区域的起始位置
        cut_x = np.random.randint(0, w - cut_w)
        cut_y = np.random.randint(0, h - cut_h)

        # 克隆新的图像和目标
        new_image1 = imagesin[idx1].clone()
        new_image2 = imagesin[idx2].clone()
        new_target1 = targetsin[idx1].clone()
        new_target2 = targetsin[idx2].clone()

        # 替换切割区域
        new_image1[:, cut_y:cut_y + cut_h, cut_x:cut_x + cut_w] = imagesin[idx2][:, cut_y:cut_y + cut_h, cut_x:cut_x + cut_w]
        new_target1[cut_y:cut_y + cut_h, cut_x:cut_x + cut_w] = targetsin[idx2][cut_y:cut_y + cut_h, cut_x:cut_x + cut_w]
        new_image2[:, cut_y:cut_y + cut_h, cut_x:cut_x + cut_w] = imagesin[idx1][:, cut_y:cut_y + cut_h, cut_x:cut_x + cut_w]
        new_target2[cut_y:cut_y + cut_h, cut_x:cut_x + cut_w] = targetsin[idx1][cut_y:cut_y + cut_h, cut_x:cut_x + cut_w]

        # 存储新的图像和目标
        output_images[i] = new_image1
        output_images[i + 1] = new_image2
        output_targets[i] = new_target1
        output_targets[i + 1] = new_target2

    return output_images, output_targets


