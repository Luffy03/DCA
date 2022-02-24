# -*- coding:utf-8 -*-

# @Filename: label2color
# @Project : Unsupervised_Domian_Adaptation
# @date    : 2021-12-04 18:39
# @Author  : Linshan

import cv2
import argparse
import os.path as osp
import torch.backends.cudnn as cudnn
import torch.optim as optim
import math
from utils.tools import *
from utils.my_tools import *
from module.Encoder import Deeplabv2
from module.my_model import *
from data.nj import NJLoader
from utils.tools import COLOR_MAP
from ever.core.iterator import Iterator
from tqdm import tqdm
from torch.nn.utils import clip_grad
import ever as er
from skimage.io import imsave, imread
from module.viz import VisualizeSegmm
parser = argparse.ArgumentParser(description='Run MY methods.')
parser.add_argument('--config_path', type=str, default='st.my.2rural',
                    help='config path')
args = parser.parse_args()
cfg = import_config(args.config_path)


def run(vis_dir):
    palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()
    viz_op = VisualizeSegmm(vis_dir, palette)

    dataloader = NJLoader(cfg.TARGET_DATA_CONFIG)

    for _, ret_gt in tqdm(dataloader):

        cls_gt = ret_gt['cls'].cpu().numpy().astype(np.int32)

        if cfg.SNAPSHOT_DIR is not None:
            for fname, label in zip(ret_gt['fname'], cls_gt):
                viz_op(label, fname.replace('tif', 'png'))


if __name__ == '__main__':
    vis_path = './LoveDA/Val/Rural/color_png'
    run(vis_path)
