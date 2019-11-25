import numpy as np
from random import uniform, randint
import tensorflow as tf
import tensorflow_addons as tfa

""" Converts scan image to cuboid by padding with zeros
"""
def cubify_scan(data: np.ndarray, cube_dim: int) -> np.ndarray:
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

@tf.function
def augment_xy(images, labels, angle_range=0.15, shift_range=10):
    angle = uniform(-angle_range, angle_range)
    shift = randint(-shift_range, shift_range)
    
    images = tfa.image.transform_ops.rotate(images, angle)
    labels = tfa.image.transform_ops.rotate(labels, angle)
    
    images = tfa.image.translate_ops.translate(images, [shift, shift])
    labels = tfa.image.translate_ops.translate(labels, [shift, shift])

    return images, labels
