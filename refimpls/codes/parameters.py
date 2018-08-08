# -*- coding: utf-8 -*-
"""define all global parameters here."""
import os
import sys
import argparse
import importlib.util
import platform
import logging
from os.path import join
import torch.distributed as dist
import json

import mlbench.models as models
from mlbench.utils.log import log, log0
from mlbench.utils.auxiliary import info2path


class ArgDict(dict):
    # TODO: type check ? here ?
    def __init__(self, *args, **kwargs):
        super(ArgDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_config(config_file=None):
    """Parse config from file and supplement with default settings. """
    ROOT_DIRECTORY = './'
    RAW_DATA_DIRECTORY = join(ROOT_DIRECTORY, 'data/')
    TRAINING_DIRECTORY = join(RAW_DATA_DIRECTORY, 'checkpoint')
    LOG_DIRECTORY = './logging'

    # TODO: Convert the configs to nested structure.
    default_config = ArgDict()
    default_config.update({
        "_metadata": {
            "annotation": "default annotation."
        },
        "data": 'cifar10',
        "data_dir": RAW_DATA_DIRECTORY,
        "use_lmdb_data": False,
        "arch": "alexnet",
        "start_epoch": 1,
        "num_epochs": 90,
        "avg_model": False,
        "reshuffle_per_epoch": False,
        "batch_size": 256,
        "base_batch_size": 64,
        "lr": 0.01,
        "lr_decay": None,
        "lr_decay_epochs": None,
        "lr_scale": False,
        "lr_warmup": False,
        "lr_warmup_size": 5,
        "lr_lars": False,
        "lr_lars_eta": 0.002,
        "lr_lars_mode": "clip",
        "momentum": 0.9,
        "use_nesterov": False,
        "weight_decay": 5e-4,
        "drop_rate": 0.0,
        "densenet_growth_rate": 12,
        "densenet_bc_mode": False,
        "densenet_compression": 0.5,
        "wideresnet_widen_factor": 4,
        "manual_seed": 6,
        "evaluate": False,
        "eval_freq": 1,
        "summary_freq": 200,
        "timestamp": None,
        "resume": None,
        "checkpoint": TRAINING_DIRECTORY,
        "checkpoint_index": None,
        "save_all_models": False,
        "save_some_models": '30,60,80',
        "log_dir": LOG_DIRECTORY,
        "plot_dir": None,
        "pretrained": False,
        "device": 'distributed',
        "num_workers": 0,
        "blocks": '2,2',
        "cuda_blocks": None,
        "world": None,
        "backend": 'mpi',
        "udf": None
    })

    if config_file is not None:
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        config = {}

    default_config.update(config)

    if default_config.timestamp is None:
        default_config.timestamp = info2path(default_config)

    config_logging()
    handle_user_defined_files(default_config.udf)
    return default_config


def get_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Training for ConvNet')
    parser.add_argument('--conf', default=None, type=str)

    # parse args.
    args = parser.parse_args()

    args = get_config(args.conf)

    return args


def handle_user_defined_files(file):
    # TODO: Revert this functionality in the future?
    if file is None:
        # log("No user defined files provided.")
        pass
    elif not os.path.exists(file):
        raise OSError("UDF `{}` not found on {}".format(file, platform.node()))
    else:

        # For illustrative purposes.
        module_name = 'udf'
        spec = importlib.util.spec_from_file_location(module_name, file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Optional; only necessary if you want to be able to import the module
        # by name later.
        sys.modules[module_name] = module


class RankFilter(logging.Filter):
    def filter(self, record):
        record.rank = dist.get_rank()
        return True


def config_logging(level=logging.DEBUG):
    # TODO : allow change of logging levels and format (say, %(module)s)
    logger = logging.getLogger('mlbench')
    logger.setLevel(level)
    logger.addFilter(RankFilter())

    formatter = logging.Formatter('%(asctime)s %(name)s %(rank)s %(levelname)s: %(message)s', "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def log_args(args):
    log0('parameters: ')
    for arg in vars(args):
        log0(("\t{:40} {:100}").format(str(arg), str(getattr(args, arg))))


if __name__ == '__main__':
    args = get_args()
