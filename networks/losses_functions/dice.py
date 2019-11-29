import tensorflow as tf


def dice_loss(y_true, y_pred, smooth=1.):
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    numerator = 2. * tf.reduce_sum(y_true_flat * y_pred_flat)
    denominator = tf.reduce_sum(y_true_flat + y_pred_flat)

    return 1. - (numerator + smooth) / (denominator + smooth)
