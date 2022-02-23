import numpy as np
from matplotlib import pyplot as plt
import os
from skimage.io import imread
import random


def run_urban_val():
    img_path = './LoveDA/Val/Urban/images_png'
    mask_path = './LoveDA/Val/Urban/color_png'
    base_path = './log/regularize/2urban/vis-URBAN1000.pth'
    ours_path = './log/multi_regularize/2urban/vis-URBAN9000.pth'
    list = os.listdir(img_path)
    random.shuffle(list)
    for i in list:
        img = imread(img_path + '/' + i)
        mask = imread(mask_path + '/' + i)
        base = imread(base_path + '/' + i)
        ours = imread(ours_path + '/' + i)

        fig, axs = plt.subplots(1, 4, figsize=(16, 12))
        axs[0].imshow((img).astype(np.uint8))
        axs[0].axis("off")

        axs[1].imshow((mask).astype(np.uint8))
        axs[1].axis("off")

        axs[2].imshow(base.astype(np.uint8))
        axs[2].axis("off")

        axs[3].imshow((ours).astype(np.uint8))
        axs[3].axis("off")

        plt.suptitle(i)
        plt.tight_layout()
        plt.show()
        plt.close()


def run_rural_val():
    img_path = './LoveDA/Val/Rural/images_png'
    mask_path = './LoveDA/Val/Rural/color_png'
    base_path = './log/multi_regularize/2rural_0.4454/vis-RURAL3000.pth'
    ours_path = './log/multi_regularize/2rural_0.4454/vis-URBAN10000.pth'
    list = os.listdir(img_path)
    random.shuffle(list)
    for i in list:
        img = imread(img_path + '/' + i)
        mask = imread(mask_path + '/' + i)
        base = imread(base_path + '/' + i)
        ours = imread(ours_path + '/' + i)

        fig, axs = plt.subplots(1, 4, figsize=(16, 12))
        axs[0].imshow((img).astype(np.uint8))
        axs[0].axis("off")

        axs[1].imshow((mask).astype(np.uint8))
        axs[1].axis("off")

        axs[2].imshow(base.astype(np.uint8))
        axs[2].axis("off")

        axs[3].imshow((ours).astype(np.uint8))
        axs[3].axis("off")

        plt.suptitle(i)
        plt.tight_layout()
        plt.show()
        plt.close()


def run_urban_test():
    img_path = './LoveDA/Test/Urban/images_png'
    base_path = './log/baseline/2urban/vis-URBAN3000.pth_TEST'
    ours_path = './log/multi_regularize/2urban/vis-URBAN10000.pth_TEST'
    list = os.listdir(img_path)
    random.shuffle(list)
    for i in list:
        img = imread(img_path + '/' + i)
        base = imread(base_path + '/' + i)
        ours = imread(ours_path + '/' + i)

        fig, axs = plt.subplots(1, 3, figsize=(16, 12))
        axs[0].imshow((img).astype(np.uint8))
        axs[0].axis("off")

        axs[1].imshow(base.astype(np.uint8))
        axs[1].axis("off")

        axs[2].imshow((ours).astype(np.uint8))
        axs[2].axis("off")

        plt.suptitle(i)
        plt.tight_layout()
        plt.show()
        plt.close()


def run_rural_test():
    img_path = './LoveDA/Test/Rural/images_png'
    base_path = './log/multi_regularize/2rural_0.4454/vis-RURAL9000.pth_TEST'
    ours_path = './log/multi_regularize/2rural_best_0.4517/vis-RURAL3000.pth_TEST'
    list = os.listdir(img_path)
    random.shuffle(list)
    for i in list:
        img = imread(img_path + '/' + i)
        base = imread(base_path + '/' + i)
        ours = imread(ours_path + '/' + i)

        fig, axs = plt.subplots(1, 3, figsize=(16, 12))
        axs[0].imshow((img).astype(np.uint8))
        axs[0].axis("off")

        axs[1].imshow(base.astype(np.uint8))
        axs[1].axis("off")

        axs[2].imshow((ours).astype(np.uint8))
        axs[2].axis("off")

        plt.suptitle(i)
        plt.tight_layout()
        plt.show()
        plt.close()


if __name__ == '__main__':
    run_rural_test()