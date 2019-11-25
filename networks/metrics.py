import tensorflow as tf
import tensorflow_addons as tfa

from networks import losses

def get(name):
    metric = dict(
        binary = tf.keras.metrics.Mean('bce_loss', dtype=tf.float32),
        dice = tf.keras.metrics.Mean('dice_loss', dtype=tf.float32),
        weighted_dice = tf.keras.metrics.Mean('w_dice_loss', dtype=tf.float32),
        accuracy = tf.keras.metrics.BinaryAccuracy('val_acc'),
        fp = tf.keras.metrics.FalsePositives(dtype=tf.float32),
        fn = tf.keras.metrics.FalseNegatives(dtype=tf.float32),
        precision = tf.keras.metrics.Precision(),
        recall = tf.keras.metrics.Recall(),
        f1score = tfa.metrics.F1Score(num_classes=1, average='micro')
    )

    return metric[name]
