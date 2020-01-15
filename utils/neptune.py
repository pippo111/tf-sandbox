from tensorflow import keras
import time


class NeptuneMonitor(keras.callbacks.Callback):
    def __init__(self, experiment, evaluation=False):
        super().__init__()

        self.exp = experiment
        self.evaluation = evaluation

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_epoch_end(self, epoch, logs=None):
        for key, value in logs.items():
            if self.evaluation:
                self.exp.log_text(log_name=key, x=str(value))
            else:
                self.exp.log_metric(log_name=key, x=epoch,
                                    y=value, timestamp=time.time())
