class CallbackManager():
    def __init__(self, model, callbacks):
        self.callbacks = callbacks
        for cb in self.callbacks:
            cb.set_model(model)

    def train_start(self):
        for cb in self.callbacks:
            cb.on_train_begin()

    def train_end(self):
        for cb in self.callbacks:
            cb.on_train_end()

    def epoch_start(self, epoch):
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch)

    def epoch_end(self, epoch, loss):
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, logs={'loss': loss})
