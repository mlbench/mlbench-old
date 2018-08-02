# -*- coding: utf-8 -*-
import os

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from mlbench.utils.log import log
from mlbench.dataset.partition_data import DataPartitioner
from mlbench.dataset.data_image.imagenet import define_imagenet
from mlbench.dataset.data_image.svhn import define_svhn


def partition_dataset(args, dataset_type='train'):
    """ Given a dataset, partition it. """
    dataset = get_dataset(args, args.data, args.data_dir, split=dataset_type)
    batch_size = args.batch_size
    world_size = args.graph.n_nodes

    # partition data.
    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    partition = DataPartitioner(args, dataset, partition_sizes)
    data_to_load = partition.use(args.graph.rank)

    if dataset_type == 'train':
        args.train_dataset_size = len(dataset)
        args.num_train_samples_per_device = len(data_to_load)
        log('  We have {} samples for {}, \
            load {} data for process (rank {}), and partition it'.format(
            len(dataset), dataset_type, len(data_to_load), args.graph.rank))
    else:
        args.val_dataset_size = len(dataset)
        args.num_val_samples_per_device = len(data_to_load)
        log('  We have {} samples for {}, \
            load {} val data for process (rank {}).'.format(
            len(dataset), dataset_type, len(data_to_load), args.graph.rank))

    # use Dataloader.
    data_type_label = (dataset_type == 'train')
    data_loader = torch.utils.data.DataLoader(
        data_to_load, batch_size=batch_size,
        shuffle=data_type_label,
        num_workers=args.num_workers, pin_memory=True, drop_last=False)

    log('we have {} batches for {} for rank {}.'.format(
        len(data_loader), dataset_type, args.graph.rank))
    return data_loader


def create_dataset(args):
    log('create {} dataset for rank {}'.format(args.data, args.graph.rank))
    train_loader = partition_dataset(args, dataset_type='train')
    val_loader = partition_dataset(args, dataset_type='test')
    return train_loader, val_loader


def get_dataset(
        args, name, datasets_path, split='train', transform=None,
        target_transform=None, download=True):
    train = (split == 'train')
    root = os.path.join(datasets_path, name)
    if not os.path.exists(root):
        os.makedirs(root)

    if name == 'cifar10' or name == 'cifar100':
        # decide normalize parameter.
        if name == 'cifar10':
            dataset_loader = datasets.CIFAR10
            normalize = transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        elif name == 'cifar100':
            dataset_loader = datasets.CIFAR100
            normalize = transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

        # decide data type.
        if train:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((32, 32), 4),
                transforms.ToTensor(),
                normalize])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                normalize])
        return dataset_loader(root=root,
                              train=train,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)
    elif name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        return datasets.MNIST(root=root,
                              train=train,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)
    elif name == 'stl10':
        return datasets.STL10(root=root,
                              split=split,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)
    elif name == 'svhn':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        return define_svhn(root=root,
                           is_train=train,
                           transform=transform,
                           target_transform=target_transform,
                           download=download)
    elif name == 'imagenet':
        root = os.path.join(datasets_path, 'lmdb') if args.use_lmdb_data \
            else datasets_path
        if train:
            root = os.path.join(root, 'train{}'.format(
                '' if not args.use_lmdb_data else '.lmdb')
            )
        else:
            root = os.path.join(root, 'val{}'.format(
                '' if not args.use_lmdb_data else '.lmdb')
            )
        return define_imagenet(root=root, flag=args.use_lmdb_data,
                               cuda=args.cuda)
    else:
        raise NotImplementedError
