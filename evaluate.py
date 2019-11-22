from networks.dataset import get_loader
from networks.model import MyModel

import config as cfg

dataset_dir = cfg.setup['dataset_dir']
valid_loader = get_loader(dataset_dir, 'valid')

for model in cfg.models:
    my_model = MyModel(
        batch_size = cfg.setup['batch_size'],
        checkpoint = model['checkpoint'],
        valid_loader = valid_loader
    )

    my_model.load_model(checkpoint = model['checkpoint'], verbose=0)

    my_model.start_evaluate()
