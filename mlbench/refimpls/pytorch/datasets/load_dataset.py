import math
import os
import sys
from PIL import Image
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from .partition_data import DataPartitioner

_DATASET_MAP = {
    "cifar10v1": "CIFAR10V1",
    "cifar10v2": "CIFAR10V2",
    "mnistv1": "MNISTV1"
}


class MNISTV1(datasets.MNIST):
    def __init__(self, root, train=True, download=False):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        super(MNISTV1, self).__init__(root=root,
                                      train=train,
                                      transform=transform,
                                      download=download)


class CIFAR10V2(datasets.CIFAR10):
    """CIFAR10 with default preprocessing.

    https://github.com/bkj/basenet/blob/49b2b61e5b9420815c64227c5a10233267c1fb14/examples/cifar10.py

    TODO: use the one in MLperf as default.
    """

    def __init__(self, root, train=True, download=False):
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
        super(CIFAR10V2, self).__init__(root=root, train=train,
                                        transform=transform,
                                        download=download)


class CIFAR10V1(datasets.CIFAR10):
    """
    * https://github.com/IamTao/dl-benchmarking/blob/master/tasks/cv/pytorch/code/dataset/create_data.py
    * https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
    """

    def __init__(self, root, train=True, download=False):
        cifar10_stats = {
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2023, 0.1994, 0.2010),
        }

        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(cifar10_stats['mean'], cifar10_stats['std']),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cifar10_stats['mean'], cifar10_stats['std']),
            ])
        super(CIFAR10V1, self).__init__(root=root, train=train,
                                        transform=transform,
                                        download=download)


def maybe_download(name, datasets_path, train=True, download=True, preprocessing_version='v1'):
    """
    Find the class with dataset name and preprocessing methods. If the dataset is not in the
    given localtion, then choose to download or not depending on `download`.
    """
    root = os.path.join(datasets_path, name)
    os.makedirs(root, exist_ok=True)

    current_module = sys.modules[__name__]
    dataset_class = _DATASET_MAP[name + preprocessing_version]

    return getattr(current_module, dataset_class)(root=root, train=train, download=download)


def partition_dataset(name, root_folder, batch_size, num_workers, rank, world_size,
                      shuffle_partition_indices, preprocessing_version, train=True, download=True, pin_memory=True):
    """ Load a partition of dataset from by the rank. """
    dataset = maybe_download(name, root_folder, train=train, download=download,
                             preprocessing_version=preprocessing_version)
    num_classes = len(dataset.classes)

    # Partition dataset and use the one corresponding to `rank`.
    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    partition = DataPartitioner(dataset, rank, shuffle_partition_indices, partition_sizes)
    data_to_load = partition.use(rank)
    num_samples_per_device = len(data_to_load)

    # create a data loader.
    data_loader = torch.utils.data.DataLoader(
        data_to_load, batch_size=batch_size, shuffle=train,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

    return {'loader': data_loader,
            'num_samples_per_device': num_samples_per_device,
            'num_batches': math.ceil(1.0 * num_samples_per_device / batch_size),
            'num_classes': num_classes}


def create_dataset(options, train=True):
    """Create a dataset and add information to options."""
    dataset = partition_dataset(name=options.dataset_name,
                                root_folder=options.root_data_dir,
                                batch_size=options.batch_size,
                                num_workers=options.num_parallel_workers,
                                rank=options.rank,
                                world_size=options.world_size,
                                shuffle_partition_indices=options.shuffle_partition_indices,
                                train=train,
                                preprocessing_version=options.preprocessing_version,
                                pin_memory=options.use_cuda)

    options.num_classes = dataset['num_classes']
    if train:
        options.train_loader = dataset['loader']
        options.train_num_samples_per_device = dataset['num_samples_per_device']
        options.train_num_batches = dataset['num_batches']
    else:
        options.val_loader = dataset['loader']
        options.val_num_samples_per_device = dataset['num_samples_per_device']
        options.val_num_batches = dataset['num_batches']

    return options
