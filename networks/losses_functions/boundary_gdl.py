from networks.losses_functions.gdl import generalized_dice_loss
from networks.losses_functions.surface import surface_loss


def boundary_gdl_loss(y_true, y_pred, alpha=1.0):
    return alpha * generalized_dice_loss(y_true, y_pred) + (1 - alpha) * surface_loss(y_true, y_pred)
