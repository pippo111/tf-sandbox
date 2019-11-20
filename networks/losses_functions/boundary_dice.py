from networks.losses_functions.dice import dice_loss
from networks.losses_functions.surface import surface_loss

def boundary_dice_loss(alpha):
    def boundary_closure(y_true, y_pred):
        return alpha * dice_loss(y_true, y_pred) + (1 - alpha) * surface_loss(y_true, y_pred)

    return boundary_closure
