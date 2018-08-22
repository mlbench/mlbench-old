import os
import math
import numpy as np
import torch
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils import log
from utils.helper import AttrDict

from .partition_data import DataPartitioner


def maybe_download(name, datasets_path, split='train', transform=None,
                   target_transform=None, download=True):
    train = (split == 'train')
    root = os.path.join(datasets_path, name)
    if not os.path.exists(root):
        os.makedirs(root)

    if name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        return datasets.MNIST(root=root,
                              train=train,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)
    elif name == 'cifar10':
        # https://github.com/IamTao/dl-benchmarking/blob/master/tasks/cv/pytorch/code/dataset/create_data.py
        # https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
        # and
        # https://github.com/bkj/basenet/blob/49b2b61e5b9420815c64227c5a10233267c1fb14/examples/cifar10.py
        # Choose different std
        cifar10_stats = {
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.247059, 0.243529, 0.261569),
        }

        if train:
            transform = transforms.Compose([
                transforms.Lambda(lambda x: np.asarray(x)),
                transforms.Lambda(lambda x: np.pad(x, [(4, 4), (4, 4), (0, 0)], mode='reflect')),
                transforms.Lambda(lambda x: Image.fromarray(x)),
                transforms.RandomCrop(32),

                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(cifar10_stats['mean'], cifar10_stats['std']),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cifar10_stats['mean'], cifar10_stats['std']),
            ])

        return datasets.CIFAR10(root=root, train=train, download=True,
                                transform=transform)

    else:
        raise NotImplementedError


def partition_dataset(name, root_folder, batch_size, num_workers, rank, world_size,
                      reshuffle_per_epoch, debug, dataset_type='train'):
    """ Given a dataset, partition it. """
    dataset = maybe_download(name, root_folder, split=dataset_type)

    data_type_label = (dataset_type == 'train')

    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    partition = DataPartitioner(dataset, rank, reshuffle_per_epoch, partition_sizes)
    data_to_load = partition.use(rank)

    num_samples_per_device = len(data_to_load)

    data_loader = torch.utils.data.DataLoader(
        data_to_load, batch_size=batch_size, shuffle=data_type_label,
        num_workers=num_workers, pin_memory=True, drop_last=False)

    return AttrDict({'loader': data_loader,
                     'num_samples_per_device': num_samples_per_device,
                     'num_batches': math.ceil(1.0 * num_samples_per_device / batch_size)})


def create_dataset(name, root_folder, batch_size, num_workers, rank, world_size,
                   reshuffle_per_epoch, debug, dataset_type='train'):
    return partition_dataset(name, root_folder, batch_size, num_workers, rank, world_size,
                             reshuffle_per_epoch, debug, dataset_type=dataset_type)
