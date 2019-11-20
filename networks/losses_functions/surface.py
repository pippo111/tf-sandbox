import numpy as np
from scipy.ndimage import distance_transform_edt as distance
import tensorflow as tf

def _calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res

def _calc_dist_map_batch(y_true):
    y_true_numpy = y_true.numpy()
    return np.array([_calc_dist_map(y)
                     for y in y_true_numpy]).astype(np.float32)

def surface_loss(y_true, y_pred):
    y_true_dist_map = tf.py_function(func=_calc_dist_map_batch,
                                     inp=[y_true],
                                     Tout=tf.float32)
    multipled = y_pred * y_true_dist_map
    return tf.math.reduce_mean(multipled)
