from tensorflow.keras.layers import ZeroPadding3D, Cropping3D
from tensorflow.keras.backend import int_shape

def calc_fit_pad(width, height, depth, n_layers):
    divider = 2 ** n_layers
    w_pad, h_pad, d_pad = 0, 0, 0

    w_rest = width % divider
    h_rest = height % divider
    d_rest = depth % divider

    if w_rest:
        w_pad = (divider - w_rest) // 2
    if h_rest:
        h_pad = (divider - h_rest) // 2
    if d_rest:
        d_pad = (divider - d_rest) // 2

    return w_pad, h_pad, d_pad

def pad_to_fit(inputs, n_layers=4):
    width = int_shape(inputs)[1]
    height = int_shape(inputs)[2]
    depth = int_shape(inputs)[3]

    w_pad, h_pad, d_pad = calc_fit_pad(width, height, depth, n_layers)

    x = ZeroPadding3D((w_pad, h_pad, d_pad))(inputs)
    return x

def crop_to_fit(inputs, outputs, n_layers=4):
    width = int_shape(inputs)[1]
    height = int_shape(inputs)[2]
    depth = int_shape(inputs)[3]

    w_pad, h_pad, d_pad = calc_fit_pad(width, height, depth, n_layers)

    x = Cropping3D((w_pad, h_pad, d_pad))(outputs)
    return x
