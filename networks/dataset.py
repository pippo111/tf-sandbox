import glob
import os
import numpy as np
import random
from joblib import Parallel, delayed

from utils.image import augment_xy

from tensorflow import keras
load_img = keras.preprocessing.image.load_img
img_to_array = keras.preprocessing.image.img_to_array


class DataSequence(keras.utils.Sequence):
    def __init__(self, x_files, y_files, batch_size=16, augment=False, shuffle=False, seed=5):
        self.x, self.y = x_files, y_files
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.seed = seed
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        res = Parallel(n_jobs=1)(delayed(self.prepare)(x, y)
                                 for x, y in zip(batch_x, batch_y))

        batch_x, batch_y = zip(*res)
        batch_x = np.array(batch_x).astype(np.float32)
        batch_y = np.array(batch_y).astype(np.float32)

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            to_shuffle = list(zip(self.x, self.y))
            random.Random(self.seed).shuffle(to_shuffle)
            self.x, self.y = zip(*to_shuffle)

    def prepare(self, x, y):
        x = img_to_array(
            load_img(x, color_mode='grayscale')) / 255
        y = img_to_array(
            load_img(y, color_mode='grayscale')) / 255

        if self.augment:
            x, y = augment_xy(x, y)

        return x, y


def get_loader(dataset_dir, dataset_type, batch_size=16, shuffle=False, augment=False, limit=None):
    x_files = sorted(
        glob.glob(os.path.join(dataset_dir, dataset_type,
                               'images/**', '*.png'), recursive=True)
    )[:limit]
    y_files = sorted(
        glob.glob(os.path.join(dataset_dir, dataset_type,
                               'labels/**', '*.png'), recursive=True)
    )[:limit]

    data_sequence = DataSequence(
        x_files, y_files, batch_size, augment, shuffle)

    return lambda: data_sequence
