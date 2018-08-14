import os
import math
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils import log

from .partition_data import DataPartitioner


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


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


def partition_dataset(name, root_folder, batch_size, num_workers, rank, world_size,
                      reshuffle_per_epoch, debug, dataset_type='train'):
    """ Given a dataset, partition it. """
    dataset = maybe_download(name, root_folder, split=dataset_type)

    data_type_label = (dataset_type == 'train')

    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    partition = DataPartitioner(dataset, rank, reshuffle_per_epoch, partition_sizes)
    data_to_load = partition.use(rank)

    num_samples_per_device = len(data_to_load)
    # log.info('There are {} samples for {}, load {} data for process (rank {}), and partition it'.format(
    #     len(dataset), dataset_type, num_samples_per_device, rank))

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
