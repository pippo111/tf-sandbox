from tensorflow import keras
import tensorflow_addons as tfa

def get(name):
  optimizer_fn = dict(
    Adam=keras.optimizers.Adam(),
    RAdam=tfa.optimizers.RectifiedAdam()
  )

  return optimizer_fn[name]
