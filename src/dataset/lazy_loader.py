from typing import Optional, Type, Callable, Dict

from torchvision.transforms import transforms

import albumentations
import torch
from torch import nn, Tensor
from torch.utils import data
from torch.utils.data import DataLoader, Subset

from albumentations.pytorch.transforms import ToTensorV2 as AlbToTensor, ToTensorV2

from dataset.DA_dataloader import ThresholdTransform, CustomPictDataset


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


class AbstractLoader:
    pass


dataset = CustomPictDataset(None, None, None, load_dir='/raid/data/DA_BrainDataset/Back')
dataset.domain_preproc('/raid/data/DA_BrainDataset/philips_15', 'philips_15')

DL_DS = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=True)


class DALoader(AbstractLoader):
    batch_size = 8
    test_batch_size = 8
    image_size = 256

    # transforms = transforms.Compose([transforms.Resize([image_size, image_size]),
    #                                              transforms.PILToTensor(),
    #                                              ThresholdTransform(thr_255=240)])

    def __init__(self, path="philips_15"):
        self.dataset = CustomPictDataset(None, None, None, load_dir='/raid/data/DA_BrainDataset/Back')
        self.dataset.domain_preproc(f'/raid/data/DA_BrainDataset/{path}', f'{path}')
        N = self.dataset.__len__()

        self.dataset_train = Subset(self.dataset, range(int(N * 0.8)))

        self.loader_train = data.DataLoader(
            self.dataset_train,
            batch_size=DALoader.batch_size,
            sampler=data_sampler(self.dataset_train, shuffle=True, distributed=False),
            drop_last=True,
            num_workers=10
        )

        self.loader_train_inf = sample_data(self.loader_train)

        self.dataset_test = Subset(self.dataset, range(int(N * 0.8), N))

        self.test_loader = data.DataLoader(
            self.dataset_test,
            batch_size=DALoader.test_batch_size,
            drop_last=False,
            num_workers=10
        )

        print("DA initialize")
        print(f'train size: {len(self.dataset_train)}, test size: {len(self.dataset_test)}')


class LazyLoader:
    saved = {}

    domain_adaptation_philips15_save: Optional[DALoader] = None
    domain_adaptation_siemens15_save: Optional[DALoader] = None
    domain_adaptation_siemens3_save: Optional[DALoader] = None

    @staticmethod
    def register_loader(cls: Type[AbstractLoader]):
        LazyLoader.saved[cls.__name__] = None

    @staticmethod
    def domain_adaptation_philips15() -> DALoader:
        if not LazyLoader.domain_adaptation_philips15_save:
            LazyLoader.domain_adaptation_philips15_save = DALoader(path="philips_15")
        return LazyLoader.domain_adaptation_philips15_save

    @staticmethod
    def domain_adaptation_siemens15() -> DALoader:
        if not LazyLoader.domain_adaptation_siemens15_save:
            LazyLoader.domain_adaptation_siemens15_save = DALoader(path="siemens15")
        return LazyLoader.domain_adaptation_siemens15_save

    @staticmethod
    def domain_adaptation_siemens3() -> DALoader:
        if not LazyLoader.domain_adaptation_siemens3_save:
            LazyLoader.domain_adaptation_siemens3_save = DALoader(path="siemens3")
        return LazyLoader.domain_adaptation_siemens3_save

