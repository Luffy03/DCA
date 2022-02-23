# -*- coding:utf-8 -*-

# @Filename: my_tools
# @Project : Unsupervised_Domian_Adaptation
# @date    : 2021-11-05 09:25
# @Author  : Linshan

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import ttach as tta
from math import *
from scipy import ndimage
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from utils.densecrf import get_crf


def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[2]
    cols_missing = target_size[1] - img.shape[3]
    padded_img = F.pad(img, (0, 0, rows_missing, cols_missing), 'constant', 0)
    return padded_img


def pre_slide(model, image, num_classes=7, tile_size=(512, 512), tta=False):
    image_size = image.shape  # bigger than (1, 3, 512, 512), i.e. (1,3,1024,1024)
    overlap = 1 / 2  # 每次滑动的重合率为1/2

    stride = ceil(tile_size[0] * (1 - overlap))  # 滑动步长:769*(1-1/3) = 513
    tile_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)  # 行滑动步数:(1024-769)/513 + 1 = 2
    tile_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)  # 列滑动步数:(2048-769)/513 + 1 = 4

    full_probs = torch.zeros((1, num_classes, image_size[2], image_size[3])).cuda()  # 初始化全概率矩阵 shape(1024,2048,19)

    count_predictions = torch.zeros((1, 1, image_size[2], image_size[3])).cuda()  # 初始化计数矩阵 shape(1024,2048,19)
    tile_counter = 0  # 滑动计数0

    for row in range(tile_rows):  # row = 0,1
        for col in range(tile_cols):  # col = 0,1,2,3
            x1 = int(col * stride)  # 起始位置x1 = 0 * 513 = 0
            y1 = int(row * stride)  # y1 = 0 * 513 = 0
            x2 = min(x1 + tile_size[1], image_size[3])  # 末位置x2 = min(0+769, 2048)
            y2 = min(y1 + tile_size[0], image_size[2])  # y2 = min(0+769, 1024)
            x1 = max(int(x2 - tile_size[1]), 0)  # 重新校准起始位置x1 = max(769-769, 0)
            y1 = max(int(y2 - tile_size[0]), 0)  # y1 = max(769-769, 0)

            img = image[:, :, y1:y2, x1:x2]  # 滑动窗口对应的图像 imge[:, :, 0:769, 0:769]
            padded_img = pad_image(img, tile_size)  # padding 确保扣下来的图像为769*769

            tile_counter += 1  # 计数加1
            # print("Predicting tile %i" % tile_counter)

            # 将扣下来的部分传入网络，网络输出概率图。
            # use softmax
            if tta is True:
                padded = tta_predict(model, padded_img)
            else:
                padded = model(padded_img)
                # padded = F.softmax(padded, dim=1)

            pre = padded[:, :, 0:img.shape[2], 0:img.shape[3]]  # 扣下相应面积 shape(769,769,19)

            count_predictions[:, :, y1:y2, x1:x2] += 1  # 窗口区域内的计数矩阵加1
            full_probs[:, :, y1:y2, x1:x2] += pre  # 窗口区域内的全概率矩阵叠加预测结果

    # average the predictions in the overlapping regions
    full_probs /= count_predictions  # 全概率矩阵 除以 计数矩阵 即得 平均概率

    return full_probs   # 返回整张图的平均概率 shape(1, 1, 1024,2048)


def predict_whole(model, image, tile_size):
    image = torch.from_numpy(image)
    interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
    x = model(image.cuda())
    x = interp(x)
    return x


def predict_multiscale(model, image, scales=[0.75, 1.0, 1.25, 1.5, 1.75, 2.0], tile_size=(512, 512)):
    """
    Predict an image by looking at it with different scales.
        We choose the "predict_whole_img" for the image with less than the original input size,
        for the input of larger size, we would choose the cropping method to ensure that GPU memory is enough.
    """
    image_size = image.shape
    image = image.data.cpu().numpy()

    full_probs = torch.zeros((1, 1, image_size[2], image_size[3])).cuda()  # 初始化全概率矩阵 shape(1024,2048,19)

    for scale in scales:
        scale = float(scale)
        print("Predicting image scaled by %f" % scale)
        scale_image = ndimage.zoom(image, (1.0, 1.0, scale, scale), order=1, prefilter=False)

        scaled_probs = predict_whole(model, scale_image, tile_size)
        full_probs += scaled_probs

    full_probs /= len(scales)

    return full_probs


def tta_predict(model, img):
    tta_transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.Rotate90(angles=[0, 90, 180, 270]),
        ])

    xs = []

    for t in tta_transforms:
        aug_img = t.augment_image(img)
        aug_x = model(aug_img)
        # aug_x = F.softmax(aug_x, dim=1)

        x = t.deaugment_mask(aug_x)
        xs.append(x)

    xs = torch.cat(xs, 0)
    x = torch.mean(xs, dim=0, keepdim=True)

    return x


def crf_predict(model, img, size=1024):
    output = model(img)

    mask = F.softmax(output, dim=1)
    mask = mask.squeeze(0).permute(1, 2, 0).cpu().numpy()
    mask = mask.astype(np.float32)

    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # img = Normalize_back(img, flag=opt.dataset)
    crf_out = get_crf(mask, img.astype(np.uint8), size=size)
    return crf_out


def mixup(s_img, s_lab, t_img, t_lab):
    s_lab, t_lab = s_lab.unsqueeze(1), t_lab.unsqueeze(1)

    batch_size = s_img.size(0)
    rand = torch.randperm(batch_size)
    lam = int(np.random.beta(0.2, 0.2) * s_img.size(2))

    new_s_img = torch.cat([s_img[:, :, 0:lam, :], t_img[rand][:, :, lam:s_img.size(2), :]], dim=2)
    new_s_lab = torch.cat([s_lab[:, :, 0:lam, :], t_lab[rand][:, :, lam:s_img.size(2), :]], dim=2)

    new_t_img = torch.cat([t_img[rand][:, :, 0:lam, :], s_img[:, :, lam:t_img.size(2), :]], dim=2)
    new_t_lab = torch.cat([t_lab[rand][:, :, 0:lam, :], s_lab[:, :, lam:t_img.size(2), :]], dim=2)

    new_s_lab, new_t_lab = new_s_lab.squeeze(1), new_t_lab.squeeze(1)

    return new_s_img, new_s_lab, new_t_img, new_t_lab



