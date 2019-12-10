import numpy as np
import os
import pandas as pd

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

from networks import network
from networks import losses
from networks import optimizers
from networks import metrics
from networks.callbacks import CallbackManager, TimerCallback, BasePrinterCallback, MetricPrinterCallback, AlphaCounterCallback
from networks.metrics import MetricManager
from utils.image import cubify_scan, augment_xy
from utils.vtk import render_mesh

AUTOTUNE = tf.data.experimental.AUTOTUNE


class MyModel():
    def __init__(
        self,
        callbacks=[],
        checkpoint_dir='output/models',
        seed=5
    ):
        tf.random.set_seed(seed)

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

        self.checkpoint_dir = checkpoint_dir

    def setup_model(
        self,
        train_generator=None,
        valid_generator=None,
        test_generator=None,
        checkpoint='checkpoint'
    ):
        """Basic model setup for generators and checkpoint filename
        """

        if train_generator:
            self.train_dataset = tf.data.Dataset.from_generator(
                train_generator, (tf.float32, tf.float32))
            # self.train_dataset = self.train_dataset.cache()
            self.train_dataset = self.train_dataset.prefetch(
                buffer_size=AUTOTUNE)

        if valid_generator:
            self.valid_dataset = tf.data.Dataset.from_generator(
                valid_generator, (tf.float32, tf.float32))
            # self.valid_dataset = self.valid_dataset.cache()
            self.valid_dataset = self.valid_dataset.prefetch(
                buffer_size=AUTOTUNE)

        if test_generator:
            self.test_dataset = tf.data.Dataset.from_generator(
                test_generator, (tf.float32, tf.float32))

        self.checkpoint = checkpoint
        self.checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint)

    def create_model(self,
                     arch='Unet',
                     optimizer_fn='RAdam',
                     loss_fn='dice',
                     n_filters=16,
                     input_shape=(256, 176),
                     verbose=1):
        """Creates model architecture and setup loss function and optimizer

        Architecture is taken from arch library
        Loss function is taken from loss functions library
        Optimizer is taken from optimizer library
        """

        self.optimizer_fn = optimizers.get(optimizer_fn)
        self.loss_name = loss_fn
        self.loss_fn = losses.get(loss_fn)

        self.model = network.get(
            name=arch,
            n_filters=n_filters,
            input_shape=input_shape
        )

        self.model.stop_training = False
        # for now we don't need to compile model as we use tensorflow 2.0 api
        # self.model.compile(optimizer=self.optimizer_fn,
        #                    loss=self.loss_fn)

        if verbose:
            self.model.summary()

    def load_model(self, verbose=1):
        """Loads model from given checkpoint
        """
        self.model = tf.keras.models.load_model(f'{self.checkpoint_path}.h5')

        if verbose:
            self.model.summary()

    def start_train(self, epochs):
        """Starts training process
        """
        if not self.train_dataset:
            raise Exception(
                'No data to process. Make sure you have setup data generators.')

        callbacks = CallbackManager(
            model=self.model,
            callbacks=[
                keras.callbacks.ModelCheckpoint(
                    f'{self.checkpoint_path}.h5', monitor='weighted_dice', save_best_only=True, verbose=1),
                # keras.callbacks.EarlyStopping(
                #     monitor='weighted_dice', mode='min', patience=25, verbose=1),
                MetricPrinterCallback(training=True),
                AlphaCounterCallback(epochs=epochs),
                TimerCallback(),
                BasePrinterCallback(epochs)
            ])

        metrics = MetricManager(
            ['weighted_dice'], training=True)

        callbacks.train_start()

        for epoch in range(epochs):
            callbacks.epoch_start(epoch)

            # train
            for step, (images, labels) in enumerate(self.train_dataset):
                callbacks.train_batch_start(step)

                loss, logits = self.train_step(
                    images, labels, self.model._cb_alpha)

                metrics.train_batch_end(labels, logits, loss)
                callbacks.train_batch_end(step)

            # validate
            for step, (images, labels) in enumerate(self.valid_dataset):
                callbacks.test_batch_start(step)

                logits = self.model(images, training=False)

                # not very clean, but don't have solution yet to pass extra params
                if self.loss_name.startswith('boundary_'):
                    loss = self.loss_fn(
                        labels, logits, alpha=self.model._cb_alpha)
                else:
                    loss = self.loss_fn(labels, logits)

                metrics.test_batch_end(labels, logits, loss)
                callbacks.test_batch_end(step)

            logs = metrics.epoch_end()
            callbacks.epoch_end(epoch, logs)

            if self.model.stop_training:
                break

        callbacks.train_end()

    def start_evaluate(self):
        threshold = 0.5

        callbacks = CallbackManager(
            model=self.model,
            callbacks=[
                MetricPrinterCallback(),
                BasePrinterCallback()
            ])

        metrics = MetricManager([
            'accuracy',
            'dice',
            'weighted_dice',
            'fp',
            'fn',
            'precision',
            'recall',
            'f1score'
        ])

        for step, (images, labels) in enumerate(self.valid_dataset):
            callbacks.test_batch_start(step)

            logits = self.model(images, training=False)

            metrics.test_batch_end(labels, logits)
            callbacks.test_batch_end(step)

        logs = metrics.epoch_end()
        callbacks.epoch_end(1, logs)

        self.save_results(logs)

    def start_visualize(self):
        scan_mask = list()

        for step, (images, labels) in enumerate(self.test_dataset):
            logits = self.model(images, training=False)
            preds = tf.dtypes.cast(logits > 0.5, tf.int8)
            scan_mask.append(preds.numpy().astype(np.int8))

        scan_mask = np.concatenate([img for img in scan_mask])
        scan_mask = scan_mask.squeeze()

        print('Mask shape:', scan_mask.shape)

        scan_mask = cubify_scan(scan_mask, 256)

        render_mesh([
            {
                'name': 'lateral_ventricles',
                'data': scan_mask,
                'color': 'Green',
                'opacity': 1.0
            }
        ], 256)

    @tf.function
    def train_step(self, images, labels, alpha=None):
        with tf.GradientTape() as tape:
            logits = self.model(images, training=True)

            # not very clean, but don't have solution yet to pass extra params
            if self.loss_name.startswith('boundary_'):
                loss = self.loss_fn(
                    labels, logits, alpha=self.model._cb_alpha)
            else:
                loss = self.loss_fn(labels, logits)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer_fn.apply_gradients(
            zip(grads, self.model.trainable_variables))

        return loss, logits

    def save_results(self, results):
        csv_file = f'{self.checkpoint_dir}/results.csv'

        formats = {
            'accuracy': '%',
            'precision': '%',
            'recall': '%',
            'f1score': '%'
        }

        for name in results:
            if name in formats and formats[name] == '%':
                results[name] = f'{results[name] * 100:0.3f}%'

        output = pd.DataFrame([{'name': self.checkpoint, **results}])

        if not os.path.exists(csv_file):
            output.to_csv(csv_file, index=False, header=True, mode='a')
        else:
            output.to_csv(csv_file, index=False, header=False, mode='a')
