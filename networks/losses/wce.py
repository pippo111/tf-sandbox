import tensorflow as tf

def weighted_binary_crossentropy_loss(weights):
    if not weights:
        weights = { 'background': 0.5, 'structure': 0.5 }

    def weighted_binary_crossentropy_coef(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # tf.mean(tf.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * weights['structure'] + (1. - y_true) * weights['background']
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return tf.reduce_mean(weighted_b_ce)

    return weighted_binary_crossentropy_coef
