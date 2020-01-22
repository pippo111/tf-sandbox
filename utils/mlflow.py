from tensorflow import keras
import mlflow

"""Log metrics to MLflow.

  Arguments:
      evaluation: Training mode
          If set to 'True' only final metrics will be logged
          If set to 'False' epoch by epoch metrics will be logged in form of chart
          
          Default: False

  Example:
  ```python
  mlflow.set_experiment('experiment')
  mlflow.start_run()
  mlflow_callback = MLflowMonitor(evaluation=True)
  mlflow.end_run()
  ```
  """


class MLflowMonitor(keras.callbacks.Callback):
    def __init__(self, evaluation=False):
        super().__init__()

        self.evaluation = evaluation

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_epoch_end(self, epoch, logs=None):
        if self.evaluation:
            mlflow.log_params(logs)
        else:
            mlflow.log_metrics(logs, step=epoch)
