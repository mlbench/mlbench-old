# -*- coding: utf-8 -*-
import platform
import os
import cv2
from shutil import copyfile

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.distributed as dist

from parameters import get_args, log_args
from mlbench.utils.log import log, log0, configure_log
from mlbench.utils.set_conf import set_conf
from mlbench.models.create_model import create_model
from mlbench.runs.distributed_running import train_and_validate as train_val_op


def main(args):
    """distributed training via mpi backend."""
    assert args.backend == 'mpi'

    # To solve an issue described in `https://github.com/pytorch/pytorch/issues/1355`
    cv2.setNumThreads(0)

    torch.set_num_threads(0)

    # init process
    dist.init_process_group('mpi')

    # set config.
    set_conf(args)
    configure_log(args)

    # enable cudnn accelerator if we are using cuda.
    if args.graph.on_gpu:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    # create model and deploy the model.
    model, criterion, optimizer = create_model(args)

    log(
        'Rank {} with block {} on {} {}-{}'.format(
            args.graph.rank,
            args.graph.rank_2_block[args.graph.rank],
            platform.node(),
            'GPU' if args.graph.on_gpu else 'CPU',
            args.graph.device
        )
    )

    # decide how to run depending on the user defined files
    if dist.get_rank() == 0:
        log_args(args)

    log0("*"*200)

    # train amd evaluate model.
    train_val_op(args, model, criterion, optimizer)

    log0("*"*200)


if __name__ == '__main__':
    args = get_args()
    main(args)
