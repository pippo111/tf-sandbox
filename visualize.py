from networks.dataset import get_loader
from networks.model import MyModel

import config as cfg

dataset_dir = cfg.setup['dataset_dir']
test_loader = get_loader(dataset_dir, 'test')

model = {
    'arch': 'Unet', 'filters': 16,
    'loss_fn': 'boundary_gdl', 'optimizer_fn': 'RAdam',
    'checkpoint': f'{cfg.setup["struct"]}_boundary_gdl'
}

my_model = MyModel(
    batch_size = cfg.setup['batch_size'],
    checkpoint = model['checkpoint'],
    test_loader = test_loader
)

my_model.load_model(checkpoint = model['checkpoint'], verbose=0)

my_model.start_visualize()
