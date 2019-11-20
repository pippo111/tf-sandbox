from networks.dataset import get_loader
from networks.model import MyModel

dataset_dir = '/home/filip/Projekty/ML/datasets/processed/mindboggle_84_coronal_176x256_lateral_ventricle_inn'
train_loader = get_loader(dataset_dir, 'train', limit=512)
valid_loader = get_loader(dataset_dir, 'valid')

my_model = MyModel(train_loader = train_loader, valid_loader = valid_loader)
my_model.start_train()
