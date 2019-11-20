import glob
import os
import numpy as np

from tensorflow import keras

from sklearn.utils import class_weight

load_img = keras.preprocessing.image.load_img
img_to_array = keras.preprocessing.image.img_to_array

def get_loader(dataset_dir, dataset_type, limit=None):
    x_files = sorted(
        glob.glob(os.path.join(dataset_dir, dataset_type, 'images/**', '*.png'), recursive=True)
    )[:limit]
    y_files = sorted(
        glob.glob(os.path.join(dataset_dir, dataset_type, 'labels/**', '*.png'), recursive=True)
    )[:limit]

    for i, (x_file, y_file) in enumerate(zip(x_files, y_files)):
        x = (img_to_array(load_img(x_file, color_mode='grayscale')) / 255).astype(np.float32)
        y = (img_to_array(load_img(y_file, color_mode='grayscale')) / 255).astype(np.float32)

dataset_dir = '/home/filip/Projekty/ML/datasets/processed/mindboggle_84_coronal_176x256_lateral_ventricle_inn'
get_loader(dataset_dir, 'train', limit=512)
