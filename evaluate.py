from networks.dataset import get_loader
from networks.model import MyModel

import config as cfg

dataset_dir = cfg.setup['dataset_dir']
valid_loader = get_loader(dataset_dir, 'valid',
                          limit=cfg.setup['valid_ds_limit'])

for model in cfg.models:
    my_model = MyModel()

    my_model.setup_model(
        valid_generator=valid_loader,
        checkpoint=model['checkpoint'])

    my_model.load_model(verbose=1)

    my_model.start_evaluate()
