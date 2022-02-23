# -*- coding:utf-8 -*-

# @Filename: densecrf
# @Project : Unsupervised_Domian_Adaptation
# @date    : 2021-11-05 09:49
# @Author  : Linshan

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile
import os
import matplotlib.pyplot as plt
#import pydensecrf.densecrf as dcrf
#from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian


def get_crf(mask, img, num_classes=7, size=512):
    mask = np.transpose(mask, (2, 0, 1))
    img = np.ascontiguousarray(img)

    unary = -np.log(mask + 1e-8)
    unary = unary.reshape((num_classes, -1))
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF2D(size, size, num_classes)
    d.setUnaryEnergy(unary)

    d.addPairwiseGaussian(sxy=5, compat=3)
    d.addPairwiseBilateral(sxy=10, srgb=13, rgbim=img, compat=10)

    output = d.inference(10)
    map = np.argmax(output, axis=0).reshape((size, size))

    return map


def run(mask_path, img_path, label_path):
    list = os.listdir(mask_path)

    for i in list:
        mask = np.load(os.path.join(mask_path, i))
        img = tifffile.imread(os.path.join(img_path, i[:-4] + '.tif'))[:, :, :3]
        label = tifffile.imread(os.path.join(label_path, i[:-4] + '.tif'))

        mask = mask.astype(np.float32)

        # get crf_output
        out = get_crf(mask, img)
        show(img, label, mask, out)


def show(img, label, mask, out):
    mask = np.argmax(mask, axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    out = np.expand_dims(out, axis=-1)

    mask, out = value_to_rgb(mask), value_to_rgb(out)
    fig, axs = plt.subplots(1, 4, figsize=(14, 4))

    axs[0].imshow(img.astype(np.uint8))
    axs[0].axis("off")

    axs[1].imshow(label.astype(np.uint8))
    axs[1].axis("off")

    axs[2].imshow(mask.astype(np.uint8))
    axs[2].axis("off")

    axs[3].imshow(out.astype(np.uint8))
    axs[3].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()
