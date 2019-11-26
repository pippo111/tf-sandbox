import glob
import os
import numpy as np

from utils.image import augment_xy

from tensorflow import keras
load_img = keras.preprocessing.image.load_img
img_to_array = keras.preprocessing.image.img_to_array


def get_loader(dataset_dir, dataset_type, augment=False, limit=None):
    x_files = sorted(
        glob.glob(os.path.join(dataset_dir, dataset_type,
                               'images/**', '*.png'), recursive=True)
    )[:limit]
    y_files = sorted(
        glob.glob(os.path.join(dataset_dir, dataset_type,
                               'labels/**', '*.png'), recursive=True)
    )[:limit]

    def loader():
        for i, (x_file, y_file) in enumerate(zip(x_files, y_files)):
            x = (img_to_array(load_img(x_file, color_mode='grayscale')) /
                 255).astype(np.float32)
            y = (img_to_array(load_img(y_file, color_mode='grayscale')) /
                 255).astype(np.float32)

            if augment:
                x, y = augment_xy(x, y)

            yield x, y

    return loader
