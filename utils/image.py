import cv2
import numpy as np
from random import uniform, randint
import tensorflow as tf
from albumentations import RandomBrightnessContrast, ShiftScaleRotate, Compose


def cubify_scan(data: np.ndarray, cube_dim: int) -> np.ndarray:
    """ Converts scan image to cuboid by padding with zeros
    """
    pad_w = (cube_dim - data.shape[0]) // 2
    pad_h = (cube_dim - data.shape[1]) // 2
    pad_d = (cube_dim - data.shape[2]) // 2

    data = np.pad(
        data,
        [(pad_w, pad_w), (pad_h, pad_h), (pad_d, pad_d)],
        mode='constant',
        constant_values=0
    )

    return data


def augment_xy(image, mask):
    augmentation = Compose([
        RandomBrightnessContrast(brightness_limit=0.15,
                                 contrast_limit=0.15, p=0.5),
        ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=10,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            value=0.,
            p=0.5
        )
    ])

    augmented = augmentation(image=image, mask=mask)

    return augmented['image'], augmented['mask']
