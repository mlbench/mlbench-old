# -*- coding: utf-8 -*-
from os.path import join, isfile

import torch
import torch.nn as nn

import mlbench.models as models
from mlbench.optim.sgd import SGD
from mlbench.utils.opfiles import remove_folder


def init_model(args):
    print("=> creating model '{}'".format(args.arch))
    if 'wideresnet' in args.arch:
        model = models.__dict__['wideresnet'](args)
    elif 'resnet' in args.arch:
        model = models.__dict__['resnet'](args)
    elif 'densenet' in args.arch:
        model = models.__dict__['densenet'](args)
    else:
        model = models.__dict__[args.arch](args)
    return model


def stat_model(args, model):
    print('Total params for process {}: {}M'.format(
        args.graph.rank,
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    ))


def define_optimizer(args, model):
    # define the param to optimize.
    params_dict = dict(model.named_parameters())
    params = [
        {
            'params': [value],
            'name': key,
            'weight_decay': args.weight_decay if 'bn' not in key else 0.0
        }
        for key, value in params_dict.items()
    ]

    # define the optimizer.
    optimizer = SGD(
        params, lr=args.learning_rate, momentum=args.momentum,
        nesterov=args.use_nesterov, args=args)
    return optimizer


def create_model(args):
    """Create model, criterion and optimizer.
    If args.graph.on_gpu is True, use ps_id as GPU_id.
    """
    model = init_model(args)
    stat_model(args, model)

    # define the criterion.
    criterion = nn.CrossEntropyLoss()

    # define the optimizer.
    optimizer = define_optimizer(args, model)

    # place model and criterion.
    if args.graph.on_gpu:
        model.cuda()
        criterion = criterion.cuda()

    # (optional) reload checkpoint
    resume_previous_status(args, model, optimizer)
    return model, criterion, optimizer


def correct_previous_resume(args, old_args):
    signal = (args.avg_model == old_args.avg_model) and \
        (args.data == old_args.data) and \
        (args.num_epochs >= old_args.num_epochs) and \
        (args.lr == old_args.lr) and \
        (args.momentum == old_args.momentum) and \
        (args.batch_size == old_args.batch_size) and \
        (args.block_size == old_args.block_size)
    print('the status of previous resume: {}'.format(signal))
    return signal


def resume_previous_status(args, model, optimizer):
    if args.resume:
        if args.checkpoint_index is not None:
            # reload model from a specific checkpoint index.
            checkpoint_index = '_epoch_' + args.checkpoint_index
        else:
            # reload model from the latest checkpoint.
            checkpoint_index = ''
        checkpoint_path = join(
            args.resume, 'checkpoint{}.pth.tar'.format(checkpoint_index))

        print('try to load previous model from the path:{}'.format(
            checkpoint_path))
        if isfile(checkpoint_path):
            print("=> loading checkpoint {} for {}".format(
                args.resume, args.graph.rank))

            # get checkpoint.
            checkpoint = torch.load(checkpoint_path)

            if not correct_previous_resume(args, checkpoint['arguments']):
                raise RuntimeError('=> the checkpoint is not correct. skip.')
            else:
                # restore some run-time info.
                args.start_epoch = checkpoint['current_epoch'] + 1
                args.local_index = checkpoint['local_index']
                args.best_prec1 = checkpoint['best_prec1']
                args.best_epoch = checkpoint['arguments'].best_epoch

                # reset path for log.
                remove_folder(args.checkpoint_root)
                args.checkpoint_root = args.resume
                args.checkpoint_dir = join(args.resume, str(args.graph.rank))
                # restore model.
                model.load_state_dict(checkpoint['state_dict'])
                # restore optimizer.
                optimizer.load_state_dict(checkpoint['optimizer'])
                # logging.
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['current_epoch']))
                return
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
