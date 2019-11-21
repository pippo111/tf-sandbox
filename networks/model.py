import numpy as np
import time
import os

import tensorflow as tf

from networks import network
from networks import losses
from networks import optimizers

class MyModel():
    def __init__(
        self,
        epochs=100,
        arch='Unet',
        optimizer_fn='RAdam',
        loss_fn='dice',
        n_filters=16,
        input_shape=(256, 176),
        batch_size=16,
        checkpoint='checkpoint',
        train_loader=None,
        valid_loader=None
    ):
        self.epochs = epochs
        self.optimizer_fn = optimizers.get(optimizer_fn)
        self.loss_name = loss_fn
        self.loss_fn = losses.get(loss_fn)
        self.batch_size = batch_size
        self.checkpoint_path = os.path.join('output/models', checkpoint)

        self.model = network.get(
            name = arch,
            optimizer_function = optimizers.get(optimizer_fn),
            loss_function = losses.get(loss_fn),
            n_filters = n_filters,
            input_shape = input_shape
        )

        if train_loader:
            self.train_dataset = tf.data.Dataset.from_generator(train_loader, (tf.float32, tf.float32))
            self.train_dataset = self.train_dataset.shuffle(1024).batch(batch_size)

        if valid_loader:
            self.valid_dataset = tf.data.Dataset.from_generator(valid_loader, (tf.float32, tf.float32))
            self.valid_dataset = self.valid_dataset.batch(batch_size)

    @tf.function
    def train_step(self, images, labels, alpha=None):
        with tf.GradientTape() as tape:
            logits = self.model(images, training=True)

            if self.loss_name.startswith('boundary_'):
                loss = self.loss_fn(alpha)(labels, logits)
            else:
                loss = self.loss_fn(labels, logits)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer_fn.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss, logits

    def train(self, epoch):
        metric_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        metric_acc = tf.keras.metrics.BinaryAccuracy('acc')

        alpha_step = 1 / self.epochs
        alpha = 1 - epoch * alpha_step

        for step, (images, labels) in enumerate(self.train_dataset):
            loss, logits = self.train_step(images, labels, alpha)
            metric_acc(labels, logits)
            metric_loss(loss)

            if step % 16 == 0:
                print(f'Train batch number: {step}...', end="\r")

        res_loss = metric_loss.result().numpy()
        res_acc = metric_acc.result().numpy()

        metric_loss.reset_states()
        metric_acc.reset_states()

        return res_loss, res_acc

    def validate(self):
        metric_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
        metric_dice_loss = tf.keras.metrics.Mean('dice_loss', dtype=tf.float32)
        metric_w_dice_loss = tf.keras.metrics.Mean('w_dice_loss', dtype=tf.float32)
        metric_val_acc = tf.keras.metrics.BinaryAccuracy('val_acc')

        for step, (images, labels) in enumerate(self.valid_dataset):
            logits = self.model(images, training=False)
            metric_val_loss(losses.get('binary')(labels, logits))
            metric_dice_loss(losses.get('dice')(labels, logits))
            metric_w_dice_loss(losses.get('weighted_dice')(labels, logits))
            metric_val_acc(labels, logits)

            if step % 16 == 0:
                print(f'Validation batch number: {step}...', end="\r")

        res_val_loss = metric_val_loss.result().numpy()
        res_dice_loss = metric_dice_loss.result().numpy()
        res_w_dice_loss = metric_w_dice_loss.result().numpy()
        res_val_acc = metric_val_acc.result().numpy()

        metric_val_loss.reset_states()
        metric_dice_loss.reset_states()
        metric_w_dice_loss.reset_states()
        metric_val_acc.reset_states()

        return res_val_loss, res_dice_loss, res_w_dice_loss, res_val_acc

    def start_train(self):
        best_result = np.Inf
        trials = 0

        self.model.summary()

        for epoch in range(self.epochs):
            start = time.time()

            loss, acc = self.train(epoch)
            val_loss, dice_loss, w_dice_loss, val_acc = self.validate()

            end = time.time()

            print(f'Train time for epoch {epoch + 1} / {self.epochs}: {end - start:.3f}s')
            print(f'Train loss: {loss:0.5f}, accuracy: {acc * 100:0.2f}%')
            print(f'Validation loss: {val_loss:0.5f}, accuracy: {val_acc * 100:0.2f}%')
            print(f'Validation dice: {dice_loss:0.5f}, weighted: {w_dice_loss:0.5f}')

            if dice_loss < best_result:
                print(f'Model improved {best_result} -> {dice_loss}')
                best_result = dice_loss
                trials = 0
                print(f'Saving checkpoint to: {self.checkpoint_path}.h5')
                self.model.save(f'{self.checkpoint_path}.h5')
                self.save_results(val_loss, dice_loss, w_dice_loss, val_acc)

            else:
                print(f'No improvements from {best_result}. Trial {trials}.')
                if trials == 12:
                    print('Early stopping')
                    break
                trials += 1

            print('-----------------------------------------------------------')

    def save_results(self, val_loss, dice_loss, w_dice_loss, val_acc):
        with open(f'{self.checkpoint_path}.csv', 'w') as f:
            f.write(f'val_loss,dice_loss,w_dice_loss,val_acc\n')
            f.write(f'{val_loss},{dice_loss},{w_dice_loss},{val_acc}')
