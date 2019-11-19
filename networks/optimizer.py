from tensorflow.keras.optimizers import Adam
from tensorflow_addons.optimizers import rectified_adam as RAdam

def get(name):
  optimizer_fn = dict(
    Adam=Adam(),
    RAdam=RAdam
  )

  return optimizer_fn[name]
