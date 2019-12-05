from networks.dataset import get_loader
from networks.model import MyModel

import config as cfg

dataset_dir = cfg.setup['dataset_dir']
train_loader = get_loader(dataset_dir, 'train',
                          augment=cfg.setup['augment'], shuffle=True)
valid_loader = get_loader(dataset_dir, 'valid')

for model in cfg.models:
    my_model = MyModel()

    my_model.setup_model(
        train_generator=train_loader,
        valid_generator=valid_loader,
        checkpoint=model['checkpoint'])

    my_model.create_model(arch=model['arch'],
                          optimizer_fn=model['optimizer_fn'],
                          loss_fn=model['loss_fn'],
                          n_filters=model['filters'],
                          input_shape=cfg.setup['input_shape'],
                          verbose=1)

    my_model.start_train(epochs=cfg.setup['epochs'])

    # my_model.create_model(
    #     epochs=cfg.setup['epochs'],
    #     input_shape=cfg.setup['input_shape'],
    #     arch=model['arch'],
    #     optimizer_fn=model['optimizer_fn'],
    #     loss_fn=model['loss_fn'],
    #     n_filters=model['filters'],
    # )

    # my_model.load_model(checkpoint=model['checkpoint'], verbose=0)

    # my_model.start_evaluate()
