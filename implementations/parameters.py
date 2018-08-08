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

import mlbench.models as models
from mlbench.utils.log import log, log0
from mlbench.utils.auxiliary import info2path


def get_args():
    ROOT_DIRECTORY = './'
    RAW_DATA_DIRECTORY = join(ROOT_DIRECTORY, 'data/')
    TRAINING_DIRECTORY = join(RAW_DATA_DIRECTORY, 'checkpoint')
    LOG_DIRECTORY = './logging'

    model_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__"))

    # feed them to the parser.
    parser = argparse.ArgumentParser(
        description='PyTorch Training for ConvNet')

    # add arguments.
    # dataset.
    parser.add_argument('--data', default='cifar10',
                        help='a specific dataset name')
    parser.add_argument('--data_dir', default=RAW_DATA_DIRECTORY,
                        help='path to dataset')
    parser.add_argument('--use_lmdb_data', default=False, type=str2bool,
                        help='use sequential lmdb dataset for better loading.')

    # model
    parser.add_argument('--arch', '-a', default='alexnet',
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: alexnet)')

    # fundamental training and learning scheme
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--num_epochs', type=int, default=90)
    parser.add_argument('--avg_model', type=str2bool, default=False)
    parser.add_argument('--reshuffle_per_epoch', default=False, type=str2bool)
    parser.add_argument('--batch_size', '-b', default=256, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--base_batch_size', default=64, type=int)

    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate')
    parser.add_argument('--lr_decay', type=str2bool, default=None)
    parser.add_argument('--lr_decay_epochs', type=str, default=None)
    parser.add_argument('--lr_scale', type=str2bool, default=False)
    parser.add_argument('--lr_warmup', type=str2bool, default=False)
    parser.add_argument('--lr_warmup_size', type=int, default=5)
    parser.add_argument('--lr_lars', type=str2bool, default=False)
    parser.add_argument('--lr_lars_eta', type=float, default=0.002)
    parser.add_argument('--lr_lars_mode', type=str, default='clip')

    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--use_nesterov', default=False, type=str2bool)
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--drop_rate', default=0.0, type=float)

    # models.
    parser.add_argument('--densenet_growth_rate', default=12, type=int)
    parser.add_argument('--densenet_bc_mode', default=False, type=str2bool)
    parser.add_argument('--densenet_compression', default=0.5, type=float)

    parser.add_argument('--wideresnet_widen_factor', default=4, type=int)

    # miscs
    parser.add_argument('--manual_seed', type=int,
                        default=6, help='manual seed')
    parser.add_argument('--evaluate', '-e', dest='evaluate',
                        type=str2bool, default=False,
                        help='evaluate model on validation set')
    parser.add_argument('--eval_freq', default=1, type=int)
    parser.add_argument('--summary_freq', default=200, type=int)
    parser.add_argument('--timestamp', default=None, type=str)

    # checkpoint
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--checkpoint', '-c', default=TRAINING_DIRECTORY,
                        type=str,
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--checkpoint_index', type=str, default=None)
    parser.add_argument('--save_all_models', type=str2bool, default=False)
    parser.add_argument('--save_some_models', type=str, default='30,60,80')
    parser.add_argument('--log_dir', default=LOG_DIRECTORY)
    parser.add_argument('--plot_dir', default=None,
                        type=str, help='path to plot the result')
    parser.add_argument('--pretrained', dest='pretrained', type=str2bool,
                        default=False, help='use pre-trained model')

    # device
    parser.add_argument('--device', type=str, default='distributed')
    parser.add_argument('--hostfile', type=str, default='hostfile')
    parser.add_argument('--mpi_path', type=str, default='/mlodata1/.openmpi')
    parser.add_argument('--on_k8s', type=str2bool, default=True)
    parser.add_argument('--python_path', type=str, default='/home/lin/.conda/envs/dmlb-env/bin/python')
    parser.add_argument('-j', '--num_workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--blocks', default='2,2', type=str,
                        help='partition processes to blocks.')
    parser.add_argument('--cuda_blocks', type=str, default=None,
                        help='if we configure it, we can use GPU.')
    parser.add_argument('--world', default=None, type=str,
                        help='number of distributed processes')
    parser.add_argument('--backend', default='mpi', type=str)
    parser.add_argument('--udf', default=None, type=str,
                        help='A path to .py file of user defined functions.')

    # parse args.
    args = parser.parse_args()
    if args.timestamp is None:
        args.timestamp = info2path(args)

    config_logging()
    handle_user_defined_files(args.udf)
    return args


def handle_user_defined_files(file):
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


def print_args(args):
    print('parameters: ')
    for arg in vars(args):
        print(arg, getattr(args, arg))


def log_args(args):
    log0('parameters: ')
    for arg in vars(args):
        log0(("\t{:40} {:100}").format(str(arg), str(getattr(args, arg))))


if __name__ == '__main__':
    args = get_args()
