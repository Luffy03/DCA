from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize, RandomCrop, RandomScale
from albumentations import OneOf, Compose
from albumentations import *
import ever as er


TARGET_SET = 'URBAN'

source_dir = dict(
    image_dir=[
        './LoveDA/Train/Rural/images_png/',
    ],
    mask_dir=[
        './LoveDA/Train/Rural/masks_png/',
    ],
)
target_dir = dict(
    image_dir=[
        './LoveDA/Val/Urban/images_png/',
    ],
    mask_dir=[
        './LoveDA/Val/Urban/masks_png/',
    ],
)

test_target_dir = dict(
    image_dir=[
        './LoveDA/Test/Urban/images_png/',
    ],
)

SOURCE_DATA_CONFIG = dict(
    image_dir=source_dir['image_dir'],
    mask_dir=source_dir['mask_dir'],
    transforms=Compose([
        RandomCrop(512, 512),
        OneOf([
            HorizontalFlip(True),
            VerticalFlip(True),
            RandomRotate90(True)
        ], p=0.75),
        Normalize(mean=(73.53223948, 80.01710095, 74.59297778),
                  std=(41.5113661,  35.66528876, 33.75830885),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()

    ]),
    CV=dict(k=10, i=-1),
    training=True,
    batch_size=8,
    num_workers=8,
)


TARGET_DATA_CONFIG = dict(
    image_dir=target_dir['image_dir'],
    mask_dir=target_dir['mask_dir'],
    transforms=Compose([
        RandomCrop(512, 512),
        OneOf([
            HorizontalFlip(True),
            VerticalFlip(True),
            RandomRotate90(True)
        ], p=0.75),
        Normalize(mean=(73.53223948, 80.01710095, 74.59297778),
                  std=(41.5113661, 35.66528876, 33.75830885),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()
    ]),
    CV=dict(k=10, i=-1),
    training=True,
    batch_size=8,
    num_workers=8,
)

EVAL_DATA_CONFIG = dict(
    image_dir=target_dir['image_dir'],
    mask_dir=target_dir['mask_dir'],
    transforms=Compose([
        Normalize(mean=(73.53223948, 80.01710095, 74.59297778),
                  std=(41.5113661, 35.66528876, 33.75830885),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()

    ]),
    CV=dict(k=10, i=-1),
    training=False,
    batch_size=1,
    num_workers=8,
)

TEST_DATA_CONFIG = dict(
    image_dir=test_target_dir['image_dir'],
    transforms=Compose([
        Normalize(mean=(73.53223948, 80.01710095, 74.59297778),
                  std=(41.5113661, 35.66528876, 33.75830885),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()

    ]),
    CV=dict(k=10, i=-1),
    training=False,
    batch_size=1,
    num_workers=8,
)
