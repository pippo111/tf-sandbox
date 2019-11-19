from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, BatchNormalization, Activation
from tensorflow.keras.layers import Conv3D, Conv3DTranspose
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import concatenate, Add

def resunet3d(input_shape, n_filters, loss_function, optimizer_function, batch_norm):
  # Convolutional block: BN -> ReLU -> Conv3x3
  def conv_block(
    inputs,
    n_filters,
    kernel_size=(3, 3, 3),
    strides=(1, 1, 1),
    activation='relu',
    batch_norm=False,
    padding='same'
  ):
    if batch_norm:
      x = BatchNormalization()(inputs)
    else:
      x = inputs

    if activation:
      x = Activation('relu')(x)

    x = Conv3D(
      filters=n_filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding
    )(x)

    return x

  inputs = Input((*input_shape, 1))

  # Encoding
  short1 = inputs
  conv1 = conv_block(inputs, n_filters, activation=None, batch_norm=False)
  conv1 = conv_block(conv1, n_filters, batch_norm=batch_norm)
  short1 = conv_block(short1, n_filters, activation=None, batch_norm=batch_norm)
  conv1 = Add()([conv1, short1])
  
  short2 = conv1
  conv2 = conv_block(conv1, n_filters*2, strides=(2, 2, 2), batch_norm=batch_norm)
  conv2 = conv_block(conv2, n_filters*2, batch_norm=batch_norm)
  short2 = conv_block(short2, n_filters*2, strides=(2, 2, 2), activation=None, batch_norm=batch_norm)
  conv2 = Add()([conv2, short2])

  short3 = conv2
  conv3 = conv_block(conv2, n_filters*4, strides=(2, 2, 2), batch_norm=batch_norm)
  conv3 = conv_block(conv3, n_filters*4, batch_norm=batch_norm)
  short3 = conv_block(short3, n_filters*4, strides=(2, 2, 2), activation=None, batch_norm=batch_norm)
  conv3 = Add()([conv3, short3])

  # Bridge
  short4 = conv3
  conv4 = conv_block(conv3, n_filters*8, strides=(2, 2, 2), batch_norm=batch_norm)
  conv4 = conv_block(conv4, n_filters*8, batch_norm=batch_norm)
  short4 = conv_block(short4, n_filters*8, strides=(2, 2, 2), activation=None, batch_norm=batch_norm)
  conv4 = Add()([conv4, short4])

  # Decoding
  up5 = Conv3DTranspose(filters=n_filters*4, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(conv4)
  up5 = concatenate([up5, conv3])
  short5 = up5
  conv5 = conv_block(up5, n_filters*4, batch_norm=batch_norm)
  conv5 = conv_block(conv5, n_filters*4, batch_norm=batch_norm)
  short5 = conv_block(short5, n_filters*4, activation=None, batch_norm=batch_norm)
  conv5 = Add()([conv5, short5])

  up6 = Conv3DTranspose(filters=n_filters*2, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(conv5)
  up6 = concatenate([up6, conv2])
  short6 = up6
  conv6 = conv_block(up6, n_filters*2, batch_norm=batch_norm)
  conv6 = conv_block(conv6, n_filters*2, batch_norm=batch_norm)
  short6 = conv_block(short6, n_filters*2, activation=None, batch_norm=batch_norm)
  conv6 = Add()([conv6, short6])

  up7 = Conv3DTranspose(filters=n_filters, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(conv6)
  up7 = concatenate([up7, conv1])
  short7 = up7
  conv7 = conv_block(up7, n_filters, batch_norm=batch_norm)
  conv7 = conv_block(conv7, n_filters, batch_norm=batch_norm)
  short7 = conv_block(short7, n_filters, activation=None, batch_norm=batch_norm)
  conv7 = Add()([conv7, short7])

  outputs = Conv3D(filters=1, kernel_size=(1, 1, 1), activation='sigmoid') (conv7)

  model = Model(inputs=[inputs], outputs=[outputs], name='ResUnet3d')
  model.compile(optimizer=optimizer_function, loss=loss_function, metrics=['accuracy'])

  return model