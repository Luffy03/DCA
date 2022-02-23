# -*- coding:utf-8 -*-

# @Filename: check_mean_std
# @Project : Unsupervised_Domian_Adaptation
# @date    : 2021-11-08 12:04
# @Author  : Linshan
import cv2
import os
import numpy as np
from utils.tools import *
from PIL import Image


def check_means(list):
    k = 0
    sum = np.zeros([3])

    for idx, i in enumerate(list):
        print(idx)
        img = Image.open(os.path.join(path, i))
        img = np.asarray(img)
        img = img.reshape(1024 * 1024, -1)
        mean = np.mean(img, axis=0)
        k += 1
        sum += mean

    means = sum / k
    print('means:', means)
    return means
    # means = [73.53223948, 80.01710095, 74.59297778]


def check_std(list, means):
    k = 0
    sum = np.zeros([3])
    for idx, i in enumerate(list):
        img = Image.open(os.path.join(path, i))
        img = np.asarray(img)
        img = img.reshape(1024 * 1024, -1)

        x = (img - means) ** 2
        k += 1
        sum += np.sum(x, axis=0)

    std = np.sqrt(sum / (k * 1024 * 1024))
    print('std:', std)


if __name__ == '__main__':
    path = './LoveDA/Val/Rural/images_png'
    list = os.listdir(path)
    means = check_means(list)
    check_std(list, means)

    # train rural
    # means = [73.53223948, 80.01710095, 74.59297778]
    # std = [41.5113661,  35.66528876, 33.75830885]

    # val urban
    # means = [71.82896878, 77.96547552, 75.14908362]
    # std: [36.9895465,  34.27826658, 33.51231337]

    # test urban
    # means = [69.48295011, 76.22691749, 74.13734415]
    # std = [35.56434404, 32.81422597, 32.11443484]




