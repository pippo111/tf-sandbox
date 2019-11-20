from tensorflow import keras

from networks.archs.unet import unet
from networks.archs.unet3d import unet3d
from networks.archs.resunet import resunet
from networks.archs.resunet3d import resunet3d

def get(
    name,
    input_shape=(48,64,64),
    n_filters=16,
    loss_function='binary_crossentropy',
    optimizer_function=keras.optimizers.Adam(),
    batch_norm=True
):
    networks = dict(
        Unet=unet,
        ResUnet=resunet,
        Unet3d=unet3d,
        ResUnet3d=resunet3d
    )
    
    return networks[name](input_shape, n_filters, loss_function, optimizer_function, batch_norm)
