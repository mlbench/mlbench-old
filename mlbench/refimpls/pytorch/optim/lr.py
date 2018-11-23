# -*- coding: utf-8 -*-
"""Scheduling Learning Rates.

.. rubric:: References

.. [ginsburg2018large] Ginsburg, Boris and Gitman, Igor and You, Yang
    Large Batch Training of Convolutional Networks with Layer-wise Adaptive Rate Scaling

.. [leslie2017cyclical] Leslie N. Smith
    Cyclical Learning Rates for Training Neural Networks

.. [goyal2017accurate] Goyal, Priya, et al.
    Accurate, large minibatch SGD: training imagenet in 1 hour.

.. [smith2017super] Smith, Leslie N., and Nicholay Topin.
    Super-Convergence: Very Fast Training of Residual Networks Using Large Learning Rates.


"""
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
import argparse
import numpy as np
import re
from bisect import bisect_left, bisect_right


def parse_batch_epoch(s, sep=',', type_=int):
    """
    Assume the input string ``s`` should have a pattern of ``epoch:1``, ``batch:50``.

    For list write ``epoch:1,10``. The numbers should be non-negative integers.
    """
    if re.match(r'epoch:[\d,]+\Z', s):
        s = s[len('epoch:'):]
        return {'epoch': type_(s) if sep not in s else [type_(i) for i in s.split(sep)]}
    elif re.match(r'batch:[\d,]+\Z', s):
        s = s[len('batch:'):]
        return {'batch': type_(s) if sep not in s else [type_(i) for i in s.split(sep)]}
    else:
        raise ValueError("The pattern should be `epoch:1`, `batch:50`, `epoch:1,2`, etc. Got {}"
                         .format(s))


class SchedulerParser(argparse.ArgumentParser):
    def __init__(self, add_help=False, multisteplr_milestones=True, multisteplr_gamma=True,
                 warmup=True, warmup_init_lr=True, warmup_linear_scaling=True, warmup_durations=True,
                 clr_cycle_length=True, clr_base_lr=True, clr_max_lr=True, clr_mode=True,
                 clr_gamma=True, clr_extra=True, sgd_lr_alpha=True, sgd_lr_beta=True, sgd_lr_gamma=True):
        super(SchedulerParser, self).__init__(add_help=add_help)

        if multisteplr_milestones:
            self.add_argument('--multisteplr_milestones', type=parse_batch_epoch,
                              default={"batch": [32000, 48000]}, metavar='<MSLRMS>',
                              help="[default: %(default)s] milestones for multistep learning rate schedule.")

        if multisteplr_gamma:
            self.add_argument('--multisteplr_gamma', type=float, default=0.1, metavar='<MSLRG>',
                              help="[default: %(default)s] Multiplicative factor of learning rate decay.")

        if warmup:
            self.add_argument("--warmup", default=False, action="store_true",
                              help="[default: %(default)s] linearly warmup learning rate before other scheduling."
                                   "For the moment, only implemented for multistep learning rate with warmup."
                                   "The warmup is used for training with more than one process.")

        if warmup_init_lr:
            warmup_init_lr_group = self.add_mutually_exclusive_group()
            warmup_init_lr_group.add_argument("--warmup_init_lr", type=float, default=0.0, metavar='<WILR>',
                                              help="[default: %(default)s] Initial learning rate before warmup.")

            warmup_init_lr_group.add_argument("--warmup_init_lr_nonscale", action='store_true', default=False,
                                              help="[default: %(default)s] Use nonscaled lr for initial warmup lr"
                                                   "for training. If this flag is true, then ignore")

        if warmup_linear_scaling:
            self.add_argument("--warmup_linear_scaling", action='store_true', default=False,
                              help="[default: %(default)s] scale the learning rate by a factor after warmup."
                                   "For linear scaling rule, this factor is the number of machines.")

        if warmup_durations:
            self.add_argument("--warmup_durations", type=parse_batch_epoch, default={'batch': 1}, metavar='<MLSRSI>',
                              help="[default: % (default)s] duration for the warmup."
                                   "The warumup should be a batch level.")

        if clr_cycle_length:
            self.add_argument("--clr_cycle_length", type=parse_batch_epoch, default='batch:2000', metavar='<CLRCL>',
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

        if clr_extra:
            self.add_argument("--clr_extra", type=float, default=0.1, metavar='<CLRE>',
                              help="[default: %(default)s] Extra number of iterations of training for one cycle.")

        if sgd_lr_alpha:
            self.add_argument("--alpha", type=float, default=100,
                              help="[default: %(default)s] Constant is used to calculate the optimal learning rate for"
                                   "SGD ( alpha / beta + t).")
        if sgd_lr_beta:
            self.add_argument("--beta", type=float, default=100,
                              help="[default: %(default)s] Constant is used to calculate the optimal learning rate for"
                                   "SGD ( alpha / beta + t).")
        if sgd_lr_gamma:
            self.add_argument("--sgd_lr_gamma", type=float, default=100,
                              help="[default: %(default)s] Constant is used to calculate the optimal learning rate for"
                                   "sparsified SGD ( gamma / (a + t) * lambda).")


def const(optimizer):
    return LambdaLR(optimizer, lr_lambda=lambda x: 1.0)


def triangular_learning_rates(optimizer, base_lr, max_lr, cycle_length, scale_fn, extra, mode):
    """Linearly scale the learning rates.

    If one cycle is applied with length smaller than the total number of iterations, then
    use small learning rate for the remaining iterations.

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

    if mode == 'one_cycle':
        def f(iterations):
            if iterations <= cycle_length:
                cycle = np.floor(1 + iterations / (2 * step_size))
                x = np.abs(iterations / step_size - 2 * cycle + 1)
                lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) * scale_fn(cycle, iterations)
            else:
                lr = base_lr * extra
            return lr / base_lr
    else:
        def f(iterations):
            cycle = np.floor(1 + iterations / (2 * step_size))
            x = np.abs(iterations / step_size - 2 * cycle + 1)
            lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) * scale_fn(cycle, iterations)
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

    smith2017super_ uses one cycle with extra epochs.
    """
    if options.lr_scheduler_level != 'batch':
        raise ValueError("The scheduler should be updated at batch level. Got {}."
                         .format(options.lr_scheduler_level))

    mode = options.clr_mode
    gamma = options.clr_gamma
    if mode in ['linear', 'triangular', 'one_cycle']:
        def scale_fn(cycle, iterations):
            return 1.
    elif mode == 'triangular2':
        def scale_fn(cycle, iterations):
            return 1 / (2. ** (cycle - 1))
    elif mode == 'exp_range':
        def scale_fn(cycle, iterations):
            return gamma ** iterations
    else:
        raise ValueError("Cycle mode {} not support.".format(mode))

    _cycle_unit, _cycle_length = options.lr_scheduler_level, options.clr_cycle_length[options.lr_scheduler_level]
    cycle_length = int(_cycle_length) if _cycle_unit == 'batch' \
        else float(_cycle_length) * options.train_num_batches

    return triangular_learning_rates(optimizer, options.clr_base_lr, options.clr_max_lr,
                                     cycle_length=cycle_length, scale_fn=scale_fn,
                                     extra=options.clr_extra,
                                     mode=mode)


def multistep_learning_rates_with_warmup(options, optimizer):
    """Use multistep learning rate schedule with warmup.

    In goyal2017accurate_, warmup is used in order to apply the ``Linear Scaling Rule``.
    Starting from the ``base_lr``, lr gradually increases to ``base_lr * scaling_factor``.
    Then use multiply the learning rate by ``gamma`` at specified milestones.

    :param options: all configs
    :type options: argparse.Namespace
    :param optimizer: optimizer associated with the scheduler
    :type optimizer: torch.nn.optim.optimizer
    :returns: a learning rate scheduler
    :rtype: LambdaLR
    :raises: ValueError, ValueError, ValueError
    """
    scaling_factor = options.world_size if options.warmup_linear_scaling else 1
    if options.warmup_init_lr_nonscale:
        lr = options.lr_per_sample * options.batch_size
    else:
        lr = options.lr

    base_lr = lr * scaling_factor

    warmup_durations = options.warmup_durations.get(options.lr_scheduler_level, 0)
    milestones = options.multisteplr_milestones[options.lr_scheduler_level]

    gamma = options.multisteplr_gamma
    warmup = options.warmup

    if options.warmup_init_lr_nonscale:
        warmup_init_lr = lr
    else:
        warmup_init_lr = options.warmup_init_lr

    if not list(milestones) == sorted(milestones):
        raise ValueError('Milestones should be a list of increasing integers.'
                         'Got {}'.format(milestones))

    if warmup_durations >= milestones[0]:
        raise ValueError("The scaling phase should be earlier than the first milestone."
                         "Got {} and {}".format(warmup_durations, milestones[0]))

    def f(durations):
        if warmup and durations <= warmup_durations:
            warmup_progress = durations / warmup_durations
            lr = warmup_progress * base_lr + (1 - warmup_progress) * warmup_init_lr
        else:
            lr = base_lr * gamma ** bisect_right(milestones, durations)
        return lr / base_lr

    for group in optimizer.param_groups:
        group['initial_lr'] = base_lr
    optimizer.base_lrs = [base_lr for _ in optimizer.param_groups]
    return LambdaLR(optimizer, lr_lambda=f)


def sgd_optimal_learning_rates(options, optimizer):
    """
    Learning rate schedule for SGD (alpha / (t + beta))
    :param options: all configs
    :param optimizer: optimizer associated with the scheduler
    """
    beta = options.beta
    alpha = options.alpha

    def f(iterations):
        return beta / (beta + iterations)

    for group in optimizer.param_groups:
        group['initial_lr'] = alpha / beta

    optimizer.base_lrs = [alpha / beta for _ in optimizer.param_groups]
    return LambdaLR(optimizer, lr_lambda=f)


def sparsified_sgd_optimal_learning_rate(options, optimizer):
    """
    Learning rate schedule for sparsifiedSGD (gamma / lambda * (t + a))
    param options: all configs
    param optimizer: optimizer associated with the scheduler
    """
    # TODO get feature size from the config file
    a = 2000 / options.sparse_grad_size
    l2_coef = options.l2_coef
    gamma = options.sgd_lr_gamma

    def f(iterations):
        return 1 / max(1, (a + iterations))

    optimizer.base_lrs = [gamma / l2_coef for _ in optimizer.param_groups]
    for group in optimizer.param_groups:
        group['initial_lr'] = gamma / l2_coef

    return LambdaLR(optimizer, lr_lambda=f)


def get_scheduler(options, optimizer):
    if options.lr_scheduler == 'const':
        return const(optimizer)
    elif options.lr_scheduler == 'CLR':
        return cyclical_learning_rates(options, optimizer)
    elif options.lr_scheduler == 'MultiStepLRW':
        return multistep_learning_rates_with_warmup(options, optimizer)
    elif options.lr_scheduler == 'sgd_optimal':
        return sgd_optimal_learning_rates(options, optimizer)
    elif options.lr_scheduler == 'sparsified_sgd':
        return sparsified_sgd_optimal_learning_rate(options, optimizer)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    pass
