# -*- coding: utf-8 -*-
from os.path import join
import torch
import platform
import torch.distributed as dist

from mlbench.utils.opfiles import build_dirs
from mlbench.utils.topology import FCGraph
from mlbench.utils.log import log, log0, config_logging


def set_checkpoint(args):
    # TODO: check
    args.checkpoint_root = join(
        args.checkpoint, args.data, args.arch,
        args.device if args.device is not None else '', args.timestamp)
    args.checkpoint_dir = join(args.checkpoint_root, str(args.graph.rank))
    args.save_some_models = args.save_some_models.split(',')

    # if the directory does not exists, create them.
    build_dirs(args.checkpoint_dir)


def set_lr(args):
    # TODO: check
    args.lr_change_epochs = [
        int(l) for l in args.lr_decay_epochs.split(',')] \
        if args.lr_decay_epochs is not None \
        else None
    args.learning_rate_per_sample = 0.1 / args.base_batch_size
    args.learning_rate = \
        args.learning_rate_per_sample * args.batch_size * args.graph.n_nodes \
        if args.lr_scale else args.lr
    args.old_learning_rate = args.learning_rate


def set_local_stat(args):
    # TODO: check
    args.local_index = 0
    args.best_prec1 = 0
    args.best_epoch = []
    args.val_accuracies = []


def set_conf(args, verbose=True):
    # init process
    dist.init_process_group(args.backend)

    # TODO: do we need manual seed for cpu?
    torch.cuda.manual_seed(args.manual_seed)

    # define the graph for the computation.
    args.graph = FCGraph(dist.get_rank(), args.blocks, args.cuda_blocks)

    if args.graph.on_gpu:
        torch.cuda.set_device(args.graph.device)

    # local conf.
    set_local_stat(args)

    # define checkpoint for logging.
    set_checkpoint(args)

    # define learning rate and learning rate decay scheme.
    set_lr(args)

    # enable cudnn accelerator if we are using cuda.
    if args.graph.on_gpu:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    # Configure the loggers.
    config_logging(args)

    if verbose:
        log_deployment(args)
        dist.barrier()
        log_args(args)


def log_args(args):
    log0('parameters: ')
    for arg in vars(args):
        log0(("\t{:40} {:100}").format(str(arg), str(getattr(args, arg))))


def log_deployment(args):
    log(
        'Rank {} with block {} on {} {}-{}'.format(
            args.graph.rank,
            args.graph.rank_2_block[args.graph.rank],
            platform.node(),
            'GPU' if args.graph.on_gpu else 'CPU',
            args.graph.device
        )
    )
