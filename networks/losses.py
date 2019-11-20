import tensorflow as tf

from networks.losses_functions.dice import dice_loss
from networks.losses_functions.wce import weighted_binary_crossentropy_loss
from networks.losses_functions.weighted_dice import weighted_dice_loss

def get(name, weights=None):
    loss_fn = dict(
        binary=tf.keras.losses.BinaryCrossentropy(),
        dice=dice_loss,
        weighted_dice=weighted_dice_loss,
        wce=weighted_binary_crossentropy_loss(weights)
    )

    return loss_fn[name]
