import numpy as np
import time
import os
import pandas as pd

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

from networks import network
from networks import losses
from networks import optimizers
from networks import metrics
from networks.callbacks import CallbackManager, TimerCallback, BasePrinterCallback
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
            self.train_dataset = self.train_dataset.cache()
            self.train_dataset = self.train_dataset.prefetch(
                buffer_size=AUTOTUNE)

        if valid_generator:
            self.valid_dataset = tf.data.Dataset.from_generator(
                valid_generator, (tf.float32, tf.float32))
            self.valid_dataset = self.valid_dataset.cache()
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
        self.model.compile(optimizer=self.optimizer_fn,
                           loss=self.loss_fn)

        if verbose:
            self.model.summary()

    def load_model(self, verbose=1):
        """Loads model from given checkpoint
        """
        self.model = tf.keras.models.load_model(f'{self.checkpoint_path}.h5')

        if verbose:
            self.model.summary()

    def start_train(self, epochs, callbacks):
        """Starts training process
        """
        if not self.train_dataset:
            raise Exception(
                'No data to process. Make sure you have setup data generators.')

        callbacks = CallbackManager(
            model=self.model,
            callbacks=[
                keras.callbacks.ModelCheckpoint(
                    f'{self.checkpoint_path}.h5', monitor='val_loss', save_best_only=True, verbose=1),
                keras.callbacks.EarlyStopping(
                    monitor='loss', mode='min', patience=2, verbose=1),
                TimerCallback(),
                BasePrinterCallback(
                    epochs,
                    steps_per_train_epoch=len(list(self.train_dataset)),
                    steps_per_test_epoch=len(list(self.valid_dataset)))
            ])

        metrics_manager = MetricManager(
            metrics=['f1score', 'fp'])

        callbacks.train_start()

        for epoch in range(epochs):
            callbacks.epoch_start(epoch)

            alpha_step = 1 / epochs
            alpha = 1 - epoch * alpha_step

            # train
            for step, (images, labels) in enumerate(self.train_dataset):
                callbacks.train_batch_start(step)

                loss, logits = self.train_step(images, labels, alpha)

                metrics_manager.train_batch_end(loss, labels, logits)
                callbacks.train_batch_end(step)

            # validate
            for step, (images, labels) in enumerate(self.valid_dataset):
                callbacks.test_batch_start(step)

                logits = self.model(images, training=False)
                loss = losses.get('weighted_dice')(labels, logits)

                metrics_manager.test_batch_end(loss, labels, logits)
                callbacks.test_batch_end(step)

            # print(f'Train loss: {loss:0.5f}, accuracy: {acc * 100:0.2f}%')
            # print(
            #     f'Validation dice: {val_loss:0.5f}, accuracy: {val_acc * 100:0.2f}%')

            logs = metrics_manager.epoch_end()
            callbacks.epoch_end(epoch, logs)

            if self.model.stop_training:
                break

        callbacks.train_end()

    def start_evaluate(self):
        start = time.time()

        results = self.evaluate()

        end = time.time()

        self.show_results(results)
        self.save_results(results)

    def start_visualize(self, test_generator):
        test_dataset = tf.data.Dataset.from_generator(
            test_generator, (tf.float32, tf.float32))

        scan_mask = list()

        for step, (images, labels) in enumerate(test_dataset):
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

            if self.loss_name.startswith('boundary_'):
                loss = self.loss_fn(alpha)(labels, logits)
            else:
                loss = self.loss_fn(labels, logits)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer_fn.apply_gradients(
            zip(grads, self.model.trainable_variables))

        return loss, logits

    def evaluate(self):
        threshold = 0.5

        metric_names = [
            'accuracy',
            'fp',
            'fn',
            'precision',
            'recall',
            'f1score'
        ]
        losses_names = [
            'binary',
            'dice',
            'weighted_dice'
        ]

        results = dict()
        histories = dict()

        for name in metric_names + losses_names:
            histories[name] = metrics.get(name)

        for step, (images, labels) in enumerate(self.valid_dataset):
            logits = self.model(images, training=False)
            preds = tf.dtypes.cast(logits > threshold, tf.float32)

            for name in metric_names:
                histories[name](labels, preds)
            for name in losses_names:
                histories[name](losses.get(name)(labels, preds))

            # if step % 4 == 0:
            #     print(f'Validation batch number: {step}...', end="\r")

        for name in metric_names + losses_names:
            results[name] = histories[name].result().numpy()
            histories[name].reset_states()

        return results

    def show_results(self, results):
        print(
            f'Validation loss: {results["binary"]}, accuracy: {results["accuracy"] * 100:0.3f}%')
        print(
            f'Validation dice: {results["dice"]}, weighted: {results["weighted_dice"]}')
        print(
            f'False positives: {results["fp"]}, false negatives: {results["fn"]}')
        print(
            f'Precision: {results["precision"]}, recall: {results["recall"]}')
        print(f'F1 Score: {results["f1score"]}')
        print('-----------------------------------------------------------')

    def save_results(self, results):
        csv_file = f'{self.checkpoint_dir}/results.csv'

        formats = {
            'name': None,
            'accuracy': '%',
            'fp': None,
            'fn': None,
            'precision': '%',
            'recall': '%',
            'f1score': '%',
            'binary': None,
            'dice': None,
            'weighted_dice': None
        }

        for name in results:
            if formats[name] == '%':
                results[name] = f'{results[name] * 100:0.3f}%'

        output = pd.DataFrame([{'name': self.checkpoint, **results}])

        if not os.path.exists(csv_file):
            output.to_csv(csv_file, index=False, header=True, mode='a')
        else:
            output.to_csv(csv_file, index=False, header=False, mode='a')
