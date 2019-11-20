import tensorflow as tf

from networks.losses_functions.dice import dice_loss
from networks.losses_functions.weighted_dice import weighted_dice_loss
from networks.losses_functions.gdl import generalized_dice_loss
from networks.losses_functions.boundary_gdl import boundary_gdl_loss

def get(name, weights=None):
    loss_fn = dict(
        binary=tf.keras.losses.BinaryCrossentropy(),
        dice=dice_loss,
        weighted_dice=weighted_dice_loss,
        gdl=generalized_dice_loss,
        boundary_gdl=boundary_gdl_loss
    )

    return loss_fn[name]
