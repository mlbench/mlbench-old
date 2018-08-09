import os

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from mlbench.utils import log


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
    else:
        raise NotImplementedError


def partition_dataset(name, root_folder, batch_size, num_workers, dataset_type='train'):
    """ Given a dataset, partition it. """
    dataset = maybe_download(name, root_folder, split=dataset_type)
    data_type_label = (dataset_type == 'train')

    log.todo('TODO: Add partitioning module.')
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=data_type_label,
        num_workers=num_workers, pin_memory=True, drop_last=False)

    return data_loader


def create_dataset(name, root_folder, batch_size, num_workers, rank):
    log.debug('create {} dataset for rank {}'.format(name, rank))

    train_loader = partition_dataset(name, root_folder, batch_size, num_workers, dataset_type='train')
    val_loader = partition_dataset(name, root_folder, batch_size, num_workers, dataset_type='test')
    return train_loader, val_loader
