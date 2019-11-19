import numpy as np
import time

import tensorflow as tf

from networks import network
from networks import losses
from networks import optimizers
from networks import metrics

class MyModel():
    def __init__(
        self,
        epochs=100,
        arch='Unet',
        optimizer_fn='Adam',
        loss_fn='binary',
        n_filters=16,
        input_shape=(256, 176),
        batch_size=16,
        train_loader=None,
        valid_loader=None,
        test_loader=None
    ):
        self.epochs = epochs
        self.optimizer_fn = optimizers.get(optimizer_fn)
        self.loss_fn = losses.get(loss_fn)

        self.model = network.get(
            name = arch,
            optimizer_function = optimizers.get(optimizer_fn),
            loss_function = losses.get(loss_fn),
            n_filters = n_filters,
            input_shape = input_shape
        )

        self.metric_acc = tf.keras.metrics.BinaryAccuracy()
        self.metric_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)

        if train_loader:
            self.train_dataset = tf.data.Dataset.from_generator(train_loader, (tf.float32, tf.float32))
            self.train_dataset = self.train_dataset.shuffle(20000).batch(batch_size)

        if valid_loader:
            self.valid_dataset = tf.data.Dataset.from_generator(valid_loader, (tf.float32, tf.float32))
            self.valid_dataset = self.valid_dataset.batch(batch_size)

        if test_loader:
            self.test_dataset = tf.data.Dataset.from_generator(test_loader, (tf.float32, tf.float32))

    @tf.function
    def train_step(self, model, optimizer_fn, loss_fn, images, labels):
        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            loss = loss_fn(labels, logits)
            self.metric_acc(labels, logits)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer_fn.apply_gradients(zip(grads, model.trainable_variables))

        return loss

    def train(self):
        for step, (images, labels) in enumerate(self.train_dataset):
            loss = self.train_step(self.model, self.optimizer_fn, self.loss_fn, images, labels)
            self.metric_loss(loss)

            print(f'Batch number: {step}...', end="\r")

        res_loss = self.metric_loss.result().numpy()
        res_acc = self.metric_acc.result().numpy()

        self.metric_loss.reset_states()
        self.metric_acc.reset_states()

        return res_loss, res_acc

    def start_train(self):
        self.model.summary()

        for epoch in range(self.epochs):
            start = time.time()

            loss, acc = self.train()

            end = time.time()

            print(f'Train time for epoch #{epoch + 1} ({int(self.optimizer_fn.iterations)} total steps): {end - start:.3f}s')
            print(f'Loss: {loss}, Acc: {acc}')
            print('-----------------------------------------------------------')

    
