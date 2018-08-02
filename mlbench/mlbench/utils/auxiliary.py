# -*- coding: utf-8 -*-
"""Auxiliary functions that support for system."""
import time
from datetime import datetime


def get_fullname(o):
    """get the full name of the class."""
    return '%s.%s' % (o.__module__, o.__class__.__name__)


def str2time(string, pattern):
    """convert the string to the datetime."""
    return datetime.strptime(string, pattern)


def determine_model_info(args):
    if 'resnet' in args.arch:
        return args.arch
    elif 'densenet' in args.arch:
        return '{}{}-{}{}'.format(
            args.arch, args.densenet_growth_rate,
            'BC-' if args.densenet_bc_mode else '',
            args.densenet_compression
        )
    elif 'wideresnet' in args.arch:
        return '{}-{}'.format(
            args.arch, args.wideresnet_widen_factor
        )


def info2path(args):
    info = '{}_{}_'.format(int(time.time()), determine_model_info(args))
    info += 'lr-{}_momentum-{}_epochs-{}_basebatchsize-{}_batchsize-{}_blocksize-{}_droprate-{}_'.format(
        args.lr,
        args.momentum,
        args.num_epochs,
        args.base_batch_size,
        args.batch_size,
        args.blocks,
        args.drop_rate
        )
    info += 'lars-{}-{}_'.format(args.lr_lars_mode, args.lr_lars_eta) \
        if args.lr_lars else ''

    return info
