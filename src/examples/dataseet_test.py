import torch
from torch.utils.data import DataLoader

from dataset.DA_dataloader import CustomPictDataset
from dataset.lazy_loader import LazyLoader

batch = next(LazyLoader.domain_adaptation_philips15().loader_train_inf)
image, mask = batch['image'], batch['mask']
print(f'{image.shape}: image.shape, {mask.shape}: mask.shape')
print(f'{image.min()}: image.min, {image.max()}: image.max,  {torch.unique(mask)}: mask.unique')


batch = next(LazyLoader.domain_adaptation_siemens3().loader_train_inf)
image, mask = batch['image'], batch['mask']
print(f'{image.shape}: image.shape, {mask.shape}: mask.shape')
print(f'{image.min()}: image.min, {image.max()}: image.max,  {torch.unique(mask)}: mask.unique')
#
batch = next(LazyLoader.domain_adaptation_siemens15().loader_train_inf)
image, mask = batch['image'], batch['mask']
print(f'{image.shape}: image.shape, {mask.shape}: mask.shape')
print(f'{image.min()}: image.min, {image.max()}: image.max,  {torch.unique(mask)}: mask.unique')

