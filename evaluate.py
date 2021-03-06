import json

from networks.dataset import get_loader
from networks.model import MyModel

with open('config.json', 'r') as cfg_file:
    config = json.load(cfg_file)

setup = config['setup']
models = config['models']
dataset_dir = setup['dataset_dir']

valid_loader = get_loader(dataset_dir, 'valid',
                          limit=setup['valid_ds_limit'])

for model in models:
    my_model = MyModel()

    my_model.setup_model(
        valid_generator=valid_loader,
        checkpoint=model['checkpoint'])

    my_model.load_model(verbose=1)

    my_model.start_evaluate()
