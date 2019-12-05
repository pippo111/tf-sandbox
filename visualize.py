import json

from networks.dataset import get_loader
from networks.model import MyModel

with open('config.json', 'r') as cfg_file:
    config = json.load(cfg_file)

setup = config['setup']
models = config['models']
dataset_dir = setup['dataset_dir']

dataset_dir = setup['dataset_dir']
test_loader = get_loader(dataset_dir, 'test')

model = {
    'arch': 'Unet', 'filters': 16,
    'loss_fn': 'boundary_gdl', 'optimizer_fn': 'RAdam',
    'checkpoint': f'{setup["struct"]}_boundary_gdl'
}

my_model = MyModel()

my_model.setup_model(
    test_generator=test_loader,
    checkpoint=model['checkpoint'])

my_model.load_model(verbose=0)

my_model.start_visualize()
