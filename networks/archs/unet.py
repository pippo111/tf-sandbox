import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate

def unet(input_shape, n_filters, loss_function, optimizer_function, batch_norm=True):
  # Convolutional block: Conv3x3 -> ReLU
  def conv_block(inputs, n_filters, kernel_size=(3, 3), activation='relu', padding='same'):
    x = Conv2D(
      filters=n_filters,
      kernel_size=kernel_size,
      padding=padding
    )(inputs)

    if batch_norm:
      x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(
      filters=n_filters,
      kernel_size=kernel_size,
      padding=padding
    )(x)

    if batch_norm:
      x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

  inputs = Input((*input_shape, 1))

  # Contracting path
  conv1 = conv_block(inputs, n_filters)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

  conv2 = conv_block(pool1, n_filters*2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

  conv3 = conv_block(pool2, n_filters*4)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

  conv4 = conv_block(pool3, n_filters*8)
  pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

  # Bridge
  conv5 = conv_block(pool4, n_filters*16)

  # Expansive path
  up6 = Conv2DTranspose(filters=n_filters*8, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv5)
  up6 = concatenate([up6, conv4])
  conv6 = conv_block(up6, n_filters*8)

  up7 = Conv2DTranspose(filters=n_filters*4, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv6)
  up7 = concatenate([up7, conv3])
  conv7 = conv_block(up7, n_filters*4)

  up8 = Conv2DTranspose(filters=n_filters*2, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv7)
  up8 = concatenate([up8, conv2])
  conv8 = conv_block(up8, n_filters*2)

  up9 = Conv2DTranspose(filters=n_filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv8)
  up9 = concatenate([up9, conv1])
  conv9 = conv_block(up9, n_filters)

  outputs = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(conv9)

  model = Model(inputs=[inputs], outputs=[outputs], name='Unet')
  # model.compile(optimizer=optimizer_function, loss=loss_function, metrics=['accuracy'])

  return model
