import json

from networks.dataset import get_loader
from networks.model import MyModel

with open('config.json', 'r') as cfg_file:
    config = json.load(cfg_file)

setup = config['setup']
models = config['models']
dataset_dir = setup['dataset_dir']

train_loader = get_loader(dataset_dir, 'train',
                          augment=setup['augment'], shuffle=True, limit=setup['train_ds_limit'])
valid_loader = get_loader(dataset_dir, 'valid',
                          limit=setup['valid_ds_limit'])

for model in models:
    my_model = MyModel()

    my_model.setup_model(
        train_generator=train_loader,
        valid_generator=valid_loader,
        checkpoint=model['checkpoint'])

    my_model.create_model(arch=model['arch'],
                          optimizer_fn=model['optimizer_fn'],
                          loss_fn=model['loss_fn'],
                          n_filters=model['filters'],
                          input_shape=tuple(setup['input_shape']),
                          verbose=1)

    my_model.start_train(epochs=setup['epochs'], callbacks=[])

    my_model.load_model(verbose=1)

    my_model.start_evaluate()
