import tensorflow as tf

def dice_coef(y_true, y_pred, smooth: int = 1) -> float:
    intersection = tf.reduce_sum(tf.math.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (
            tf.reduce_sum(tf.square(y_true), -1) + tf.reduce_sum(tf.square(y_pred), -1) + smooth)


def dice_loss(y_true, y_pred) -> float:
    return 1. - dice_coef(y_true, y_pred)
