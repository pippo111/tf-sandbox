import tensorflow as tf

from networks.losses_functions.dice import dice_loss
from networks.losses_functions.surface import surface_loss

bce_loss = tf.keras.losses.BinaryCrossentropy()

def boundary_bce_loss(alpha):
    def boundary_closure(y_true, y_pred):
        return alpha * bce_loss(y_true, y_pred) + (1 - alpha) * surface_loss(y_true, y_pred)

    return boundary_closure
