import numpy as np
import time
import os
import pandas as pd

import tensorflow as tf
import tensorflow_addons as tfa

from networks import network
from networks import losses
from networks import optimizers
from networks import metrics
from utils.image import cubify_scan, augment_xy
from utils.vtk import render_mesh

AUTOTUNE = tf.data.experimental.AUTOTUNE


class MyModel():
    def __init__(
        self,
        batch_size=16,
        checkpoint='checkpoint',
        checkpoint_dir='output/models',
        train_loader=None,
        valid_loader=None,
        test_loader=None,
        seed=5
    ):
        tf.random.set_seed(seed)

        self.checkpoint = checkpoint
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
        self.batch_size = batch_size

        if train_loader:
            self.train_dataset = tf.data.Dataset.from_generator(
                lambda: train_loader, (tf.float32, tf.float32))
            self.train_dataset = self.train_dataset.cache()
            self.train_dataset = self.train_dataset.prefetch(
                buffer_size=AUTOTUNE)

        if valid_loader:
            self.valid_dataset = tf.data.Dataset.from_generator(
                lambda: valid_loader, (tf.float32, tf.float32))
            self.valid_dataset = self.valid_dataset.prefetch(
                buffer_size=AUTOTUNE)

        if test_loader:
            self.test_dataset = tf.data.Dataset.from_generator(
                lambda: test_loader, (tf.float32, tf.float32))

    def create_model(
        self,
        epochs=100,
        arch='Unet',
        optimizer_fn='RAdam',
        loss_fn='dice',
        n_filters=16,
        input_shape=(256, 176)
    ):
        self.epochs = epochs
        self.optimizer_fn = optimizers.get(optimizer_fn)
        self.loss_name = loss_fn
        self.loss_fn = losses.get(loss_fn)

        self.model = network.get(
            name=arch,
            n_filters=n_filters,
            input_shape=input_shape
        )

        self.model.summary()

    def load_model(self, checkpoint='checkpoint', checkpoint_dir='output/models', verbose=1):
        self.checkpoint = checkpoint
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint)

        self.model = tf.keras.models.load_model(f'{self.checkpoint_path}.h5')

        if verbose:
            self.model.summary()

    def start_train(self):
        best_result = np.Inf
        trials = 0

        for epoch in range(self.epochs):
            start = time.time()

            alpha_step = 1 / self.epochs
            alpha = 1 - epoch * alpha_step

            metric_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
            metric_acc = tf.keras.metrics.BinaryAccuracy('acc')

            for step, (images, labels) in enumerate(self.train_dataset):
                loss, logits = self.train_step(images, labels, alpha)
                metric_acc(labels, logits)
                metric_loss(loss)

                if step % 4 == 0:
                    print(f'Train batch number: {step}...', end="\r")

            loss = metric_loss.result().numpy()
            acc = metric_acc.result().numpy()

            metric_loss.reset_states()
            metric_acc.reset_states()

            val_loss, val_acc = self.validate(alpha)

            end = time.time()

            print(
                f'Train time for epoch {epoch + 1} / {self.epochs}: {end - start:.3f}s')
            print(f'Train loss: {loss:0.5f}, accuracy: {acc * 100:0.2f}%')
            print(
                f'Validation dice: {val_loss:0.5f}, accuracy: {val_acc * 100:0.2f}%')

            if val_loss < best_result:
                print(f'Model improved {best_result} -> {val_loss}')
                best_result = val_loss
                trials = 0
                print(f'Saving checkpoint to: {self.checkpoint_path}.h5')
                self.model.save(f'{self.checkpoint_path}.h5')

            else:
                print(f'No improvements from {best_result}. Trial {trials}.')
                if trials == 25:
                    print('Early stopping')
                    break
                trials += 1

            print('-----------------------------------------------------------')

    def start_evaluate(self):
        start = time.time()

        results = self.evaluate()

        end = time.time()

        self.show_results(results)
        self.save_results(results)

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

            if self.loss_name.startswith('boundary_'):
                loss = self.loss_fn(alpha)(labels, logits)
            else:
                loss = self.loss_fn(labels, logits)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer_fn.apply_gradients(
            zip(grads, self.model.trainable_variables))

        return loss, logits

    def validate(self, alpha):
        metric_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
        metric_val_acc = tf.keras.metrics.BinaryAccuracy('val_acc')

        for step, (images, labels) in enumerate(self.valid_dataset):
            logits = self.model(images, training=False)

            metric_val_loss(losses.get('weighted_dice')(labels, logits))
            metric_val_acc(labels, logits)

            if step % 4 == 0:
                print(f'Validation batch number: {step}...', end="\r")

        res_val_loss = metric_val_loss.result().numpy()
        res_val_acc = metric_val_acc.result().numpy()

        metric_val_loss.reset_states()
        metric_val_acc.reset_states()

        return res_val_loss, res_val_acc

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

            if step % 4 == 0:
                print(f'Validation batch number: {step}...', end="\r")

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
