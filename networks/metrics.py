import tensorflow as tf
import tensorflow_addons as tfa

from networks import losses


def get(name):
    metric = dict(
        acc=tf.keras.metrics.BinaryAccuracy('acc'),
        val_acc=tf.keras.metrics.BinaryAccuracy('val_acc'),
        binary=tf.keras.metrics.Mean('bce_loss', dtype=tf.float32),
        dice=tf.keras.metrics.Mean('dice_loss', dtype=tf.float32),
        weighted_dice=tf.keras.metrics.Mean('weighted_dice', dtype=tf.float32),
        fp=tf.keras.metrics.FalsePositives(name='fp', dtype=tf.float32),
        fn=tf.keras.metrics.FalseNegatives(name='fn', dtype=tf.float32),
        precision=tf.keras.metrics.Precision(name='precision'),
        recall=tf.keras.metrics.Recall(name='recall'),
        f1score=tfa.metrics.F1Score(
            name='f1score', num_classes=1, average='micro')
    )

    return metric[name]


class MetricManager():
    def __init__(self, metrics):
        self.logs = {}
        # loss, acc, val_loss, val_acc
        self.std_metrics = {
            'loss': tf.keras.metrics.Mean('loss', dtype=tf.float32),
            'val_loss': tf.keras.metrics.Mean('val_loss', dtype=tf.float32),
            'acc': tf.keras.metrics.BinaryAccuracy('acc'),
            'val_acc': tf.keras.metrics.BinaryAccuracy('val_acc')
        }

        self.metrics = [get(name) for name in metrics]

    def train_batch_end(self, loss, labels, logits):
        self.std_metrics['loss'](loss)
        self.std_metrics['acc'](labels, logits)

        for metric in self.metrics:
            metric(labels, logits)

    def test_batch_end(self, loss, labels, logits):
        self.std_metrics['val_loss'](loss)
        self.std_metrics['val_acc'](labels, logits)

        for metric in self.metrics:
            metric(labels, logits)

    def epoch_end(self):
        for name, metric in self.std_metrics.items():
            res = metric.result().numpy()

            self.logs.update({name: res})
            metric.reset_states()

        for metric in self.metrics:
            res = metric.result().numpy()
            name = metric.name

            self.logs.update({name: res})
            metric.reset_states()

        return self.logs
