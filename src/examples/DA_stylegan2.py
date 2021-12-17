import argparse
import sys, os

from parameters.run import RuntimeParameters

sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../../gans/'))
import torch
from dataset.lazy_loader import LazyLoader

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    parents=[
        RuntimeParameters(),
    ]
)
args = parser.parse_args()
for k in vars(args):
    print(f"{k}: {vars(args)[k]}")

device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

batch = next(LazyLoader.domain_adaptation_philips15().loader_train_inf)
image, mask = batch['image'].to(device), batch['mask'].to(device)

print(f'{image.shape}: image.shape, {mask.shape}: mask.shape')
print(f'{image.min()}: image.min, {image.max()}: image.max,  {torch.unique(mask)}: mask.unique')

batch = next(LazyLoader.domain_adaptation_siemens3().loader_train_inf)
image, mask = batch['image'].to(device), batch['mask'].to(device)

print(f'{image.shape}: image.shape, {mask.shape}: mask.shape')
print(f'{image.min()}: image.min, {image.max()}: image.max,  {torch.unique(mask)}: mask.unique')
