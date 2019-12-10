import tensorflow as tf
import tensorflow_addons as tfa

from networks.losses_functions.dice import dice_coef
from networks.losses_functions.weighted_dice import weighted_dice_coef


def get(name):
    metric = dict(
        accuracy=tf.keras.metrics.BinaryAccuracy('accuracy'),
        dice=DiceScore('dice', dtype=tf.float32),
        weighted_dice=WeightedDiceScore('weighted_dice', dtype=tf.float32),
        fp=tf.keras.metrics.FalsePositives(name='fp', dtype=tf.float32),
        fn=tf.keras.metrics.FalseNegatives(name='fn', dtype=tf.float32),
        precision=tf.keras.metrics.Precision(name='precision'),
        recall=tf.keras.metrics.Recall(name='recall'),
        f1score=tfa.metrics.F1Score(
            name='f1score', num_classes=1, average='micro')
    )

    return metric[name]


class MetricManager():
    def __init__(self, metrics=[], training=False):
        self.logs = {}
        self.training = training

        self.std_metrics = {
            'loss': tf.keras.metrics.Mean('loss', dtype=tf.float32),
            'val_loss': tf.keras.metrics.Mean('val_loss', dtype=tf.float32),
            'acc': tf.keras.metrics.BinaryAccuracy('acc'),
            'val_acc': tf.keras.metrics.BinaryAccuracy('val_acc')
        }

        self.metrics = [get(name) for name in metrics]

    def train_batch_end(self, labels, logits, loss):
        self.std_metrics['loss'](loss)
        self.std_metrics['acc'](labels, logits)

    def test_batch_end(self, labels, logits, loss=None):
        if self.training:
            self.std_metrics['val_loss'](loss)

        self.std_metrics['val_acc'](labels, logits)

        for metric in self.metrics:
            metric(labels, logits)

    def epoch_end(self):
        if self.training:
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


class DiceScore(tf.keras.metrics.Metric):

    def __init__(self, name='dice', **kwargs):
        super(DiceScore, self).__init__(name=name, **kwargs)
        self.avg_score = tf.keras.metrics.Mean('dice', dtype=tf.float32)

    def update_state(self, y_true, y_pred):
        loss = dice_coef(y_true, y_pred)
        self.avg_score(loss)

    def result(self):
        return self.avg_score.result()

    def reset_states(self):
        self.avg_score.reset_states()


class WeightedDiceScore(tf.keras.metrics.Metric):

    def __init__(self, name='weighted_dice', **kwargs):
        super(WeightedDiceScore, self).__init__(name=name, **kwargs)
        self.avg_score = tf.keras.metrics.Mean(
            'weighted_dice', dtype=tf.float32)

    def update_state(self, y_true, y_pred):
        loss = weighted_dice_coef(y_true, y_pred)
        self.avg_score(loss)

    def result(self):
        return self.avg_score.result()

    def reset_states(self):
        self.avg_score.reset_states()
