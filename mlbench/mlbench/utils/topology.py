# -*- coding: utf-8 -*-

from itertools import groupby
from functools import reduce

from abc import abstractmethod, ABCMeta
import numpy as np


class UndirectedGraph(metaclass=ABCMeta):

    @property
    @abstractmethod
    def n_nodes(self):
        pass

    @property
    @abstractmethod
    def world(self):
        pass

    @property
    @abstractmethod
    def rank(self):
        pass

    @property
    @abstractmethod
    def ranks(self):
        pass

    @property
    @abstractmethod
    def rank_2_block(self):
        pass

    @property
    @abstractmethod
    def block_2_ranks(self):
        pass

    @property
    @abstractmethod
    def device(self):
        pass

    @property
    @abstractmethod
    def on_gpu(self):
        pass

    @abstractmethod
    def get_neighborhood(self, node_id):
        pass


class FCGraph(UndirectedGraph):
    def __init__(self, rank, blocks, cuda_blocks):
        self.cur_rank = rank
        self.blocks = [int(l) for l in blocks.split(',')]
        self.cuda_blocks = \
            [int(l) for l in cuda_blocks.split(',')] \
            if cuda_blocks is not None else []

    @property
    def n_nodes(self):
        # it evaluates number of processes.
        return sum(self.blocks)

    @property
    def world(self):
        # it assigns the gpu id from 0 to n-1 for each block.
        return reduce(
            lambda a, b: a + b, [list(range(b)) for b in self.blocks])

    @property
    def rank(self):
        return self.cur_rank

    @property
    def ranks(self):
        return list(range(self.n_nodes))

    @property
    def rank_2_block(self):
        # we can map each rank id to each block id.
        return self._map_rank_to_block()

    def _map_rank_to_block(self):
        blocks = []
        for block_ind, block_size in enumerate(self.blocks):
            blocks += [block_ind] * block_size
        return dict(list(zip(self.ranks, blocks)))

    @property
    def block_2_ranks(self):
        return dict(
            [(k, [l[0] for l in g])
             for k, g in groupby(self.rank_2_block.items(),
                                 lambda x: x[1])]
        )

    @property
    def device(self):
        return self.world[self.rank]

    @property
    def on_gpu(self):
        return self._get_device_type()

    def get_neighborhood(self):
        """it will return a list of ranks that are connected with this node."""
        return self.block_2_ranks[self.rank_2_block[self.rank]]

    def _get_device_type(self):
        """ detect the device type.

        If there is no specified information, then we will use GPU by default.
        Otherwise, we will first assign the GPU location, and then CPU.
        """
        return False
        # TODO: confusing, consult Tao
        if len(self.cuda_blocks) == 0:
            return True

        num_cuda_block = len(self.blocks)
        cur_block = self.rank_2_block[self.rank]

        # safety check if args.cuda_blocks is correct.
        assert num_cuda_block == len(self.blocks)

        # detect the type of device for current rank.
        max_num_gpus_in_cur_block = self.cuda_blocks[cur_block]
        ranks_in_cur_block = self.block_2_ranks[cur_block]
        index_of_cur_rank_in_cur_block = ranks_in_cur_block.index(self.rank)

        # return the detected device type. If it is GPU, then return True.
        if index_of_cur_rank_in_cur_block + 1 <= max_num_gpus_in_cur_block:
            return True
        else:
            return False


if __name__ == '__main__':
    graph = FCGraph(rank=0, blocks='1', cuda_blocks='1')

    print(graph.world)
    print(graph.ranks)
    print(graph.device)
    print(graph.on_gpu)
    print(graph.get_neighborhood())
