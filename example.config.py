# Models common setup
setup = {
    'dataset_dir': '/home/filip/Projekty/ML/datasets/processed/mindboggle_84_coronal_176x256_lateral_ventricle_inn',
    'struct': 'lateral_ventricle',
    'epochs': 100,
    'batch_size': 16,
    'input_shape': (256, 176),
    'augment': True
}

# Model different parameters
models = [
    {
        'arch': 'Unet', 'filters': 16,
        'loss_fn': 'binary', 'optimizer_fn': 'RAdam',
        'checkpoint': f'{setup["struct"]}_binary'
    },
    {
        'arch': 'Unet', 'filters': 16,
        'loss_fn': 'dice', 'optimizer_fn': 'RAdam',
        'checkpoint': f'{setup["struct"]}_dice'
    },
    {
        'arch': 'Unet', 'filters': 16,
        'loss_fn': 'boundary_dice', 'optimizer_fn': 'RAdam',
        'checkpoint': f'{setup["struct"]}_boundary_dice'
    },
    {
        'arch': 'Unet', 'filters': 16,
        'loss_fn': 'boundary_gdl', 'optimizer_fn': 'RAdam',
        'checkpoint': f'{setup["struct"]}_boundary_gdl'
    }
]
