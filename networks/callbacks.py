import time
import datetime
from datetime import timedelta
import tensorflow as tf
from tensorflow import keras


class CallbackManager():
    def __init__(self, model, callbacks):
        self.callbacks = callbacks
        for cb in self.callbacks:
            cb.set_model(model)

    def train_start(self):
        for cb in self.callbacks:
            cb.on_train_begin()

    def train_end(self):
        for cb in self.callbacks:
            cb.on_train_end()

    def epoch_start(self, epoch):
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch)

    def epoch_end(self, epoch, loss):
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, logs={'loss': loss})

    def batch_start(self, batch):
        for cb in self.callbacks:
            cb.on_train_batch_begin(batch)

    def batch_end(self, batch, loss):
        for cb in self.callbacks:
            cb.on_train_batch_end(batch, logs={'loss': loss})


class TimerCallback(keras.callbacks.Callback):
    def __init__(self):
        self.batch_avg_metric = keras.metrics.Mean(
            'timer_avg_batch', dtype=tf.float32)
        self.epoch_avg_metric = keras.metrics.Mean(
            'timer_avg_epoch', dtype=tf.float32)

        self.batch_time = 0
        self.epoch_time = 0
        self.train_time = 0

    def on_train_begin(self):
        self.total = time.time()

    def on_train_end(self):
        self.total = time.time() - self.total

        epoch_avg = self.epoch_avg_metric.result().numpy()
        self.epoch_avg_metric.reset_states()

        total_time_s = int(round(self.total))

        print(
            f'Total time: {str(timedelta(seconds=total_time_s))}, time per epoch: {epoch_avg:.2f}s')

    def on_epoch_begin(self, epoch):
        self.epoch_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_time = time.time() - self.epoch_time
        self.epoch_avg_metric(self.epoch_time)

        batch_avg = self.batch_avg_metric.result().numpy()
        self.batch_avg_metric.reset_states()

        print(
            f'Epoch time: {self.epoch_time:.2f}s, time per batch: {batch_avg:.2f}s')

    def on_train_batch_begin(self, batch):
        self.batch_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        self.batch_time = time.time() - self.batch_time
        self.batch_avg_metric(self.batch_time)


class PrinterCallback(keras.callbacks.Callback):
    def __init__(self, epochs, steps_per_epoch):
        self.epochs = epochs
        self.steps = steps_per_epoch

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_epoch_begin(self, epoch):
        print(f'Epoch {epoch + 1} / {self.epochs}')

    def on_epoch_end(self, epoch, logs=None):
        print(f'--------------------------------------------------------------------------------------------------')

    def on_train_batch_begin(self, batch):
        print(f'Train batch {batch + 1} / {self.steps}', end='\r')

    def on_train_batch_end(self, batch, logs=None):
        pass
