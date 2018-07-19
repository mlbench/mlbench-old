import argparse
import torch
import os
import torch.distributed as dist

if __name__ == '__main__':
    dist.init_process_group('mpi')
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    vector = [0] * world_size
    vector[rank] = 1
    vector = torch.DoubleTensor(vector)

    # dist.all_reduce(vector, op=dist.reduce_op.SUM)
    print("<br> Rank {} : {}".format(rank, vector))
