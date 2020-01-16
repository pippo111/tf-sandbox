import json
import os
import neptune

from networks.dataset import get_loader, calculate_hash
from networks.model import MyModel
from utils.neptune import NeptuneMonitor

with open('config.json', 'r') as cfg_file:
    config = json.load(cfg_file)


setup = config['setup']
models = config['models']
dataset_dir = setup['dataset_dir']

train_loader = get_loader(dataset_dir, 'train',
                          augment=setup['augment'],
                          shuffle=True,
                          limit=setup['train_ds_limit'])
valid_loader = get_loader(dataset_dir, 'valid',
                          limit=setup['valid_ds_limit'])

train_hash = calculate_hash(dataset_dir, 'train', verbose=1)
valid_hash = calculate_hash(dataset_dir, 'valid', verbose=1)

# Setup Neptune.ai
os.environ['NEPTUNE_API_TOKEN'] = config['neptune']['api_key']
neptune.init(config['neptune']['project_name'])
tags = ['hippocampus', 'coronal']

for model in models:
    params = {
        'arch': model['arch'],
        'batch_size': setup['batch_size'],
        'filters': model['filters'],
        'loss_fn': model['loss_fn'],
        'optimizer_fn': model['optimizer_fn'],
        'data_augment': setup['augment']
    }

    with neptune.create_experiment(name=setup['struct'], params=params) as neptune_exp:
        neptune_exp.append_tag(tags)
        neptune_exp.log_text('train_data_version', train_hash)
        neptune_exp.log_text('valid_data_version', valid_hash)

        my_model = MyModel(checkpoint_dir=setup['checkpoint_dir'])

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

        my_model.start_train(epochs=setup['epochs'], custom_callbacks=[
            NeptuneMonitor(experiment=neptune_exp, evaluation=False)])

        my_model.load_model(verbose=1)

        my_model.start_evaluate(custom_callbacks=[
            NeptuneMonitor(experiment=neptune_exp, evaluation=True)])

        neptune_exp.log_artifact(os.path.join(
            setup['checkpoint_dir'], f"{model['checkpoint']}.h5"))
