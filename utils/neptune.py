from tensorflow import keras
import time

"""Log metrics to Neptune.ai hub. If your notebook is not configured, api key
  will be grabbed from environment variable 'NEPTUNE_API_TOKEN'

  Arguments:
      experiment: Experiment object created by Neptune to log data to, more:
          https://docs.neptune.ai/neptune-client/docs/project.html#neptune.projects.Project.create_experiment

      evaluation: Training mode
          If set to 'True' only final metrics will be logged
          If set to 'False' epoch by epoch metrics will be logged in form of chart
          
          Default: False

  Example:
  ```python
  experiment = neptune.create_experiment()
  neptune_callback = NeptuneMonitor(experiment, evaluation=True)
  ```
  """


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
