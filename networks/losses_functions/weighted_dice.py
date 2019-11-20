import tensorflow as tf
import numpy as np

def _calc_weights(y_true):
    nonzeros = tf.math.count_nonzero(y_true, dtype=tf.float32)
    total = tf.size(y_true, out_type=tf.float32)

    background = total - nonzeros
    structure = nonzeros

    background_weights = structure / total
    structure_weights = 1 - background_weights

    return { 'background': background_weights, 'structure': structure_weights }

def _calculate_partial_result(y_true, y_pred, weight) -> float:
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.math.reduce_sum(y_true_f * y_pred_f)
    return ((2 * intersection + 1) / (tf.math.reduce_sum(y_true_f) + tf.math.reduce_sum(y_pred_f) + 1) *
            weight)


def weighted_dice_coef(y_true, y_pred) -> float:
    weights = _calc_weights(y_true)

    result = 0.0
    result += _calculate_partial_result(y_true, y_pred, weights['structure'])

    neg_y_true = tf.ones(shape=tf.shape(y_true)) - y_true
    neg_y_pred = tf.ones(shape=tf.shape(y_pred)) - y_pred

    result += _calculate_partial_result(neg_y_true, neg_y_pred, weights['background'])

    return result

def weighted_dice_loss(y_true, y_pred) -> float:
    return 1. - weighted_dice_coef(y_true, y_pred)
