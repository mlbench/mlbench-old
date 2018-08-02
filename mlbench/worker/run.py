# -*- coding: utf-8 -*-
import platform
import cv2

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.distributed as dist

from parameters import get_args, log_args
from mlbench.utils.log import log, configure_log
from mlbench.utils.set_conf import set_conf
from mlbench.models.create_model import create_model
from mlbench.runs.distributed_running import train_and_validate as train_val_op


def main(args):
    """distributed training via mpi backend."""
    # init process
    dist.init_process_group('mpi')

    # set config.
    set_conf(args)

    # enable cudnn accelerator if we are using cuda.
    if args.graph.on_gpu:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    # create model and deploy the model.
    model, criterion, optimizer = create_model(args)

    # # config and report.
    configure_log(args)
    log_args(args)
    log(
        'Rank {} with block {} on {} {}-{}'.format(
            args.graph.rank,
            args.graph.rank_2_block[args.graph.rank],
            platform.node(),
            'GPU' if args.graph.on_gpu else 'CPU',
            args.graph.device
        )
    )
    print('+'*80)

    # train amd evaluate model.
    train_val_op(args, model, criterion, optimizer)


if __name__ == '__main__':
    args = get_args()
    main(args)
