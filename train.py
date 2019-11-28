from networks.dataset import get_loader
from networks.model import MyModel

import config as cfg

dataset_dir = cfg.setup['dataset_dir']
train_loader = get_loader(dataset_dir, 'train',
                          augment=cfg.setup['augment'], shuffle=True)
valid_loader = get_loader(dataset_dir, 'valid')

for model in cfg.models:
    my_model = MyModel(
        batch_size=cfg.setup['batch_size'],
        checkpoint=model['checkpoint'],
        train_loader=train_loader,
        valid_loader=valid_loader
    )

    my_model.create_model(
        epochs=cfg.setup['epochs'],
        input_shape=cfg.setup['input_shape'],
        arch=model['arch'],
        optimizer_fn=model['optimizer_fn'],
        loss_fn=model['loss_fn'],
        n_filters=model['filters'],
    )

    my_model.start_train()

    my_model.create_model(
        epochs=cfg.setup['epochs'],
        input_shape=cfg.setup['input_shape'],
        arch=model['arch'],
        optimizer_fn=model['optimizer_fn'],
        loss_fn=model['loss_fn'],
        n_filters=model['filters'],
    )

    my_model.load_model(checkpoint=model['checkpoint'], verbose=0)

    my_model.start_evaluate()
