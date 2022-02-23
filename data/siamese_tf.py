# -*- coding:utf-8 -*-

# @Filename: siamese_tf
# @Project : Unsupervised_Domian_Adaptation
# @date    : 2021-12-09 16:41
# @Author  : Linshan
import random
import math
import numpy as np
import numbers
import collections
import torch
from PIL import Image, ImageFilter

import torchvision.transforms as tf
import torchvision.transforms.functional as F


class RandGaussianBlur(object):

    # Note: there is significant difference
    # with OpenCV implementation how sigma is
    # computed for the given radius

    def __init__(self, radius=[.1, 2.]):
        self.radius = radius

    def __call__(self, image):
        radius = random.uniform(self.radius[0], self.radius[1])
        image = image.filter(ImageFilter.GaussianBlur(radius))

        return image


class MaskRandGreyscale(object):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, image):
        if self.p > random.random():
            image = F.to_grayscale(image, num_output_channels=3)

        return image


class MaskRandJitter(object):
    # Note: there is significant difference
    # with OpenCV implementation how sigma is
    # computed for the given radius

    def __init__(self, jitter=0.4, p=0.5):
        self.p = p
        self.jitter = tf.ColorJitter(brightness=jitter, \
                                     contrast=jitter, \
                                     saturation=jitter, \
                                     hue=min(0.1, jitter))

    def __call__(self, image):
        if random.random() < self.p:
            image = self.jitter(image)

        return image


def normalize(img):
    means = [71.82896878, 77.96547552, 75.14908362]
    stds = [36.9895465,  34.27826658, 33.51231337]
    return (img - means)/stds


def normalize_back(img):
    means = [71.82896878, 77.96547552, 75.14908362]
    stds = [36.9895465,  34.27826658, 33.51231337]
    return img * stds + means


def aug_img(inputs):
    images = inputs.clone()
    trans = tf.Compose([MaskRandGreyscale(),
                        MaskRandJitter(),
                        RandGaussianBlur()])
    for i, img in enumerate(images):
        img = img.permute(1, 2, 0).cpu().numpy()
        img = normalize_back(img)

        img = Image.fromarray(img.astype(np.uint8))
        for t in trans.transforms:
            img = t(img)

        img = np.asarray(img)
        img = normalize(img)
        img = img.astype(np.float32).transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        images[i] = img

    return images.to(inputs.device)