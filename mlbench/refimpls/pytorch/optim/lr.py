# -*- coding: utf-8 -*-
"""Scheduling Learning Rates.

.. rubric:: References

.. [ginsburg2018large] Ginsburg, Boris and Gitman, Igor and You, Yang
    Large Batch Training of Convolutional Networks with Layer-wise Adaptive Rate Scaling

.. [leslie2017cyclical] Leslie N. Smith
    Cyclical Learning Rates for Training Neural Networks

"""
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
import argparse
import numpy as np


class SchedulerParser(argparse.ArgumentParser):
    def __init__(self, add_help=False, multisteplr_milestones=True, multisteplr_gamma=True,
                 clr_cycle_length=True, clr_base_lr=True, clr_max_lr=True, clr_mode=True,
                 clr_gamma=True):
        super(SchedulerParser, self).__init__(add_help=add_help)

        if multisteplr_milestones:
            self.add_argument('--multisteplr_milestones', type=str, default='0.5,0.67', metavar='<MSLRMS>',
                              help="[default: %(default)s] milestones in terms of percentage of total epochs/batches.")

        if multisteplr_gamma:
            self.add_argument('--multisteplr_gamma', type=float, default=0.1, metavar='<MSLRG>',
                              help="[default: %(default)s] Multiplicative factor of learning rate decay.")

        if clr_cycle_length:
            self.add_argument("--clr_cycle_length", type=str, default='batch:2000', metavar='<CLRCL>',
                              help="[default: %(default)s] cycle length in a cyclical learning rates training."
                              "It can be `batch:int_batches` or `epoch:float_epochs`.")

        if clr_base_lr:
            self.add_argument("--clr_base_lr", type=float, default=0.001, metavar='<CLRBLR>',
                              help="[default: %(default)s] minimum and initial learning rate in cyclical"
                              "learning rates training.")
        if clr_max_lr:
            self.add_argument("--clr_max_lr", type=float, default=0.1, metavar='<CLRMLR>',
                              help="[default: %(default)s] maximum learning rate in cyclical"
                              "learning rates training. Note this maximum value might not be reached "
                              "depending on the chosen scaling mode.")

        if clr_mode:
            self.add_argument("--clr_mode", type=str, default='triangular', metavar='<CLRM>',
                              help="[default: %(default)s] scaling mode in cyclical learning rate schedule.")

        if clr_gamma:
            self.add_argument("--clr_gamma", type=float, default=0.99, metavar='<CLRG>',
                              help="[default: %(default)s] constant in 'exp_range' scaling function"
                              " in cyclical learning rate schedule.")


def const(optimizer):
    return LambdaLR(optimizer, lr_lambda=lambda x: 1.0)


def linear_learning_rates(optimizer, base_lr, max_lr, cycle_length, scale_fn):
    """Linearly scale the learning rates.

    :param optimizer: an optimizer whose learning rate is scheduled.
    :type optimizer: torch.nn.optim.optimizer
    :param base_lr: lower bound and initial lr in a cycle.
    :type base_lr: float
    :param max_lr: upper bound in a cycle
    :type max_lr: float
    :param cycle_length: length of a cycle in terms of batches.
    :type cycle_length: int
    :param scale_fn: custom scaling policy defined by a single argument lambda function, defaults to None
    :type scale_fn: callable, optional
    :returns: a learning rate scheduler
    :rtype: LambdaLR
    """
    step_size = cycle_length / 2

    def f(iterations):
        cycle = np.floor(1 + iterations / (2 * step_size))
        x = np.abs(iterations/step_size - 2 * cycle + 1)
        lr = base_lr + (max_lr-base_lr) * np.maximum(0, (1-x)) * scale_fn(cycle, iterations)
        return lr / base_lr

    # Use base_lr to overwrite the --lr
    for group in optimizer.param_groups:
        group['initial_lr'] = base_lr
    optimizer.base_lrs = [base_lr for _ in optimizer.param_groups]
    return LambdaLR(optimizer, lr_lambda=f)


def cyclical_learning_rates(options, optimizer):
    """
    Since leslie2017cyclical_ mentioned that traingular, Welch, Hann windows produce equivalent results,
    we only implement triangular learning rate policy, also known as **linear cycle**.

    The original implementation of leslie2017cyclical_ can be found from `here <https://github.com/bckenstler/CLR>`_.
    """
    if options.lr_scheduler_level != 'batch':
        raise ValueError("The scheduler should be updated at batch level. Got {}."
                         .format(options.lr_scheduler_level))

    mode = options.clr_mode
    gamma = options.clr_gamma
    if mode in ['linear', 'triangular']:
        def scale_fn(cycle, iterations): return 1.
    elif mode == 'triangular2':
        def scale_fn(cycle, iterations): return 1 / (2. ** (cycle - 1))
    elif mode == 'exp_range':
        def scale_fn(cycle, iterations): return gamma ** iterations
    else:
        raise ValueError("Cycle mode {} not support.".format(mode))

    _cycle_unit, _cycle_length = options.clr_cycle_length.split(":")
    cycle_length = int(_cycle_length) if _cycle_unit == 'batch' \
        else float(_cycle_length) * options.train_num_batches

    return linear_learning_rates(optimizer, options.clr_base_lr, options.clr_max_lr,
                                 cycle_length=cycle_length,
                                 scale_fn=scale_fn)


def get_scheduler(options, optimizer):
    if options.lr_scheduler == 'const':
        return const(optimizer)
    elif options.lr_scheduler == 'MultiStepLR':
        steps = options.train_epochs if options.lr_scheduler_level == 'epoch' \
            else options.train_num_batches * options.train_epochs

        milestone_percents = options.multisteplr_milestones.split(',')
        milestones = [int(steps * float(ms)) for ms in milestone_percents]
        return MultiStepLR(optimizer, milestones=milestones,
                           gamma=options.multisteplr_gamma)
    elif options.lr_scheduler == 'CLR':
        return cyclical_learning_rates(options, optimizer)
    else:
        raise NotImplementedError


# def adjust_learning_rate(args, optimizer, init_lr=0.1):
#     """Sets the learning rate to the initial LR decayed by the number of accessed sample.
#     """
#     # functions.
#     def define_lr_decay_by_epoch(args, epoch_index):
#         """ decay based on the number of accessed samples per device. """
#         for ind, change_epoch in enumerate(args.lr_change_epochs):
#             if epoch_index <= change_epoch:
#                 return args.learning_rate * (0.1 ** ind)
#         return args.learning_rate * (0.1 ** 3)

#     def define_lr_decay_by_index_poly(args, pow=2):
#         """ decay the learning rate polynomially. """
#         return args.learning_rate * (
#             1 - args.local_index / args.num_batches_total_train) ** 2

#     # adjust learning rate.
#     if args.lr_decay_epochs is not None:
#         num_accessed_samples = args.local_index * args.batch_size
#         epoch_index = num_accessed_samples // args.num_train_samples_per_device
#         lr = define_lr_decay_by_epoch(args, epoch_index)
#     else:
#         lr = define_lr_decay_by_index_poly(args)

#     # lr warmup at the first few epochs.
#     if args.lr_warmup and args.local_index < args.num_warmup_samples:
#         lr = (lr - init_lr) / args.num_warmup_samples * args.local_index + init_lr

#     # assign learning rate.
#     if args.old_learning_rate != lr:
#         args.old_learning_rate = lr
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr


# def adjust_learning_rate_by_lars(args, global_lr, para):
#     """Adjust the learning rate via Layer-Wise Adaptive Rate Scaling (LARS).

#     The dataset is stored in [ginsburg2018large]_.
#     """
#     w = para.data
#     g = para.grad.data

#     local_lr = w.norm() / (w.norm() + beta * g.norm())
#     # v  = m * v +

#     lr = global_lr

#     if args.lr_lars:
#         local_lr = args.lr_lars_eta * para.data.norm() / para.grad.data.norm()
#         if args.lr_lars_mode == 'clip':
#             lr = min(local_lr, lr)
#         elif args.lr_lars_mode == 'scale':
#             lr = local_lr * lr
#         else:
#             raise ValueError('Invalid LARS mode: %s' % args.lr_lars_factor)
#     return lr
