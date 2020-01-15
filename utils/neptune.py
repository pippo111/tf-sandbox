import neptune
from tensorflow import keras
import glob
import hashlib
import os
import time


class NeptuneMonitor(keras.callbacks.Callback):
    def __init__(self, project_name, params, dataset_dir='', tags=[]):
        super().__init__()
        self.params = params
        self.dataset_dir = dataset_dir
        self.tags = tags

        neptune.init(project_name)

    def on_train_begin(self):
        self.exp = neptune.create_experiment(params=self.params)
        self.exp.append_tag('hippocampus', 'coronal', 'paperspace')

        if self.dataset_dir:
            train_hash = self.calculate_hash(
                self.dataset_dir, 'train', verbose=1)
            valid_hash = self.calculate_hash(
                self.dataset_dir, 'valid', verbose=1)
            self.exp.log_text('train_data_version', train_hash)
            self.exp.log_text('valid_data_version', valid_hash)

    def on_train_end(self):
        neptune.stop()

    def on_epoch_end(self, epoch, logs=None):
        for key, value in logs.items():
            print('neptune', key, value)
            self.exp.log_metric(log_name=key, x=epoch,
                                y=value, timestamp=time.time())
            self.exp.log_text(log_name=key, x=str(
                value), timestamp=time.time())

    # Moved this function here to be sure to check md5 the same way for every experiment
    def calculate_hash(self, dataset_dir, dataset_type, file_type='png', verbose=0):
        if verbose:
            print(f'Calculating dataset {dataset_type} hash...')

        filenames = sorted(
            glob.glob(os.path.join(dataset_dir, dataset_type,
                                   '**', f'*.{file_type}'), recursive=True))

        hash_md5 = hashlib.md5()

        for filename in filenames:
            with open(filename, "rb") as f:
                for chunk in iter(lambda: f.read(2 ** 20), b""):
                    hash_md5.update(chunk)

        ds_hash = hash_md5.hexdigest()

        if verbose:
            print(f'Calculated hash: {ds_hash}')

        return ds_hash
