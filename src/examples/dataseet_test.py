from typing import List

import torch
from torch.utils.data import DataLoader

from dataset.DA_dataloader import CustomPictDataset
from dataset.lazy_loader import LazyLoader

# batch = next(LazyLoader.domain_adaptation_philips15().loader_train_inf)
# image, mask = batch['image'], batch['mask']
# print(f'{image.shape}: image.shape, {mask.shape}: mask.shape')
# print(f'{image.min()}: image.min, {image.max()}: image.max,  {torch.unique(mask)}: mask.unique')
# #

# batch = next(LazyLoader.domain_adaptation_ge3().loader_train_inf)
# image, mask = batch['image'], batch['mask']
# print(f'{image.shape}: image.shape, {mask.shape}: mask.shape')
# print(f'{image.min()}: image.min, {image.max()}: image.max,  {torch.unique(mask)}: mask.unique')
from examples.style_progressive import Noise2Style, ConditionalStyleTransform
from gan.nn.stylegan.components import EqualLinear
from nn.progressiya.base import Progressive, ProgressiveWithoutState

# batch = next(LazyLoader.celeba().loader)
# print(batch.shape)

# batch = next(LazyLoader.metfaces().loader)
# print(batch.shape)


model = Noise2Style()
image_generation = ConditionalStyleTransform()
print (image_generation(model.forward(batch_size=8), cond=torch.zeros(8, dtype=torch.int64).cuda()).shape)
#
# batch = next(LazyLoader.domain_adaptation_siemens15().loader_train_inf)
# image, mask = batch['image'], batch['mask']
# print(f'{image.shape}: image.shape, {mask.shape}: mask.shape')
# print(f'{image.min()}: image.min, {image.max()}: image.max,  {torch.unique(mask)}: mask.unique')

