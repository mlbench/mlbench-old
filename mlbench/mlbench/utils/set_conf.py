# -*- coding: utf-8 -*-
from os.path import join
import torch
import torch.distributed as dist

from mlbench.utils.opfiles import build_dirs
from mlbench.utils.topology import FCGraph


def check_args(args):
    # check the value of args.
    if 'imagenet' == args.data:
        if 'ILSVRC' not in args.data_dir:
            raise 'your should provide a correct data dir \
                that can point to imagenet'


def set_checkpoint(args):
    args.checkpoint_root = join(
        args.checkpoint, args.data, args.arch,
        args.device if args.device is not None else '', args.timestamp)
    args.checkpoint_dir = join(args.checkpoint_root, str(args.graph.rank))
    args.save_some_models = args.save_some_models.split(',')

    # if the directory does not exists, create them.
    build_dirs(args.checkpoint_dir)


def set_lr(args):
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
    args.local_index = 0
    args.best_prec1 = 0
    args.best_epoch = []
    args.val_accuracies = []


def set_conf(args):
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
