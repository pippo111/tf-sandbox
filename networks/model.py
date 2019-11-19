import numpy as np

import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy

from networks import network
from networks import loss
from networks import optimizer

@tf.function
def compute_loss(logits, labels):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))

# @tf.function
def compute_accuracy(logits, labels):
    predictions = tf.nn.softmax(logits)
    return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))

# @tf.function
def train_one_step(model, optimizer, x, y):

    with tf.GradientTape() as tape:
        logits = model(x)
        loss = binary_crossentropy(logits, y)

    # compute gradient
    grads = tape.gradient(loss, model.trainable_variables)
    # update to weights
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    accuracy = compute_accuracy(logits, y)

    # loss and accuracy is scalar tensor
    return loss, accuracy

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
        self.optimizer_function = optimizer.get(optimizer_fn)

        self.model = network.get(
            name = arch,
            optimizer_function = optimizer.get(optimizer_fn),
            loss_function = loss.get(loss_fn),
            n_filters = n_filters,
            input_shape = input_shape
        )

        if train_loader:
            self.train_dataset = tf.data.Dataset.from_generator(train_loader, (tf.float32, tf.float32))
            self.train_dataset = self.train_dataset.shuffle(20000).batch(batch_size)

        if valid_loader:
            self.valid_dataset = tf.data.Dataset.from_generator(valid_loader, (tf.float32, tf.float32))
            self.valid_dataset = self.valid_dataset.batch(batch_size)

        if test_loader:
            self.test_dataset = tf.data.Dataset.from_generator(test_loader, (tf.float32, tf.float32))

    def train(self):
        self.model.summary()

        loss = 0.0
        accuracy = 0.0
        for step, (x, y) in enumerate(self.train_dataset):
            print('Step:', step)
            loss, accuracy = train_one_step(self.model, self.optimizer_function, x, y)
            print(np.mean(loss), np.mean(accuracy))

        return loss, accuracy
