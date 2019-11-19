from networks.losses.dice import dice_loss
from networks.losses.wce import weighted_binary_crossentropy_loss


def get(name, weights=None):
    loss_fn = dict(
        binary='binary_crossentropy',
        dice=dice_loss,
        wce=weighted_binary_crossentropy_loss(weights)
    )

    return loss_fn[name]
