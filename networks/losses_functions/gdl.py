import numpy as np
import tensorflow as tf

def _calculate_weight(y_true_f, epsilon) -> float:
    target_sum = tf.reduce_sum(y_true_f)
    return 1. / tf.clip_by_value(
        target_sum * target_sum, clip_value_min=epsilon, clip_value_max=np.inf)


def generalized_dice_coef(y_true, y_pred, epsilon=1e-5) -> float:
    result = 0.0

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    weight = _calculate_weight(y_true_f, epsilon)

    intersection = tf.reduce_sum(y_true_f * y_pred_f) * weight
    denominator = tf.reduce_sum(y_pred_f + y_true_f) * weight

    neg_y_true = tf.ones(shape=tf.shape(y_true)) - y_true
    neg_y_pred = tf.ones(shape=tf.shape(y_pred)) - y_pred

    y_true_f = tf.reshape(neg_y_true, [-1])
    y_pred_f = tf.reshape(neg_y_pred, [-1])

    weight = _calculate_weight(y_true_f, epsilon)
    intersection += tf.reduce_sum(y_true_f * y_pred_f) * weight

    denominator += tf.reduce_sum(y_pred_f + y_true_f) * weight

    return 2. * intersection / tf.clip_by_value(
        denominator, clip_value_min=epsilon, clip_value_max=np.inf)


def generalized_dice_loss(y_true, y_pred) -> float:
    return 1. - generalized_dice_coef(y_true, y_pred)
