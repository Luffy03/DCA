import os
import numpy as np
import cv2
from skimage.io import imread, imsave


def run():
    img_path = '/home/hnu2/WLS/Unsupervised_Domian_Adaptation/LoveDA/Val/Rural/images_png'
    save_path = '/home/hnu2/WLS/Unsupervised_Domian_Adaptation/LoveDA/Val/Rural/images_png_new'
    path = '/home/hnu2/WLS/Unsupervised_Domian_Adaptation/log/multi_regularize/2rural/pseudo_label'
    list = os.listdir(path)
    num = 0
    for i in list:
        label = cv2.imread(os.path.join(path, i))[:, :, 0] - 1

        count = (label == 0).astype(np.uint8)
        pro = count.sum() / (1024*1024)

        if pro < 0.3:
            num += 1
            print(num)
            img = imread(os.path.join(img_path, i))
            imsave(os.path.join(save_path, i), img)


if __name__ == '__main__':
    run()


