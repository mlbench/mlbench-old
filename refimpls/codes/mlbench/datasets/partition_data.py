# -*- coding: utf-8 -*-
import random

import numpy as np
import torch
import torch.distributed as dist


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, indices):
        self.data = data
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        data_idx = self.indices[index]
        return self.data[data_idx]


class Partitioner(object):
    def consistent_indices(self, indices):
        if self.args.graph.rank == 0 and self.args.reshuffle_per_epoch:
            random.shuffle(indices)

        # broadcast.
        indices = torch.IntTensor(indices)
        group = dist.new_group(self.args.graph.ranks)
        dist.broadcast(indices, src=0, group=group)
        return list(indices)


class DataPartitioner(Partitioner):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, args, data, sizes=[0.7, 0.2, 0.1]):
        # prepare info.
        self.args = args
        self.data = data
        self.data_size = len(self.data)
        self.partitions = []

        # get shuffled/unshuffled data.
        indices = [x for x in range(0, self.data_size)]
        indices = self.consistent_indices(indices)

        # partition indices.
        sizes = np.cumsum(sizes)
        from_index = 0
        for ind, frac in enumerate(sizes):
            to_index = int(sizes[ind] * self.data_size)
            self.partitions.append(indices[from_index: to_index])
            from_index = to_index

    def use(self, partition_ind):
        return Partition(self.data, self.partitions[partition_ind])
