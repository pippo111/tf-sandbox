from networks.dataset import get_loader
from networks.model import MyModel

import config as cfg

dataset_dir = cfg.setup['dataset_dir']
train_loader = get_loader(dataset_dir, 'train')
valid_loader = get_loader(dataset_dir, 'valid')

for model in cfg.models:
    my_model = MyModel(
        epochs = cfg.setup['epochs'],
        batch_size = cfg.setup['batch_size'],
        input_shape = cfg.setup['input_shape'],
        arch = model['arch'],
        optimizer_fn = model['optimizer_fn'],
        loss_fn = model['loss_fn'],
        n_filters = model['filters'],
        checkpoint = model['checkpoint'],
        train_loader = train_loader,
        valid_loader = valid_loader
    )

    my_model.start_train()
