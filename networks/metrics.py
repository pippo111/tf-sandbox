import tensorflow as tf

def get(name):
    metric_fn = dict(
        acc=tf.keras.metrics.BinaryAccuracy()
    )

    return metric_fn[name]
