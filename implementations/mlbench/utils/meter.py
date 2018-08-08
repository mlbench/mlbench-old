# -*- coding: utf-8 -*-
import shutil
from os.path import join
import torch
from mlbench.utils.log import log


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_model_status(iter, gpu_id, model, is_weight=False, is_gradient=True):
    model_parameters = list(model.parameters())
    param = model_parameters[0]
    if is_weight:
        log("iter:{}, check process {}'s weights for 1st variable:{}".format(
            iter, gpu_id, torch.norm(param.data)))
    if is_gradient:
        log("iter:{}, check process {}'s gradients for 1st variable:{}".format(
            iter, gpu_id, torch.norm(param.grad.data)))


def define_local_tracker():
    batch_time = AverageMeter()
    data_time = AverageMeter()
    sync_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    load_time = AverageMeter()
    tracker = {
        'batch_time': batch_time,
        'data_time': data_time,
        'sync_time': sync_time,
        'load_time': load_time,
        'losses': losses,
        'top1': top1,
        'top5': top5,
        'others': {}}
    return tracker


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, dirname, filename, save_all=False):
    # save full state.
    args = state['arguments']
    checkpoint_path = join(dirname, filename)
    best_model_path = join(dirname, 'model_best.pth.tar')
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_model_path)
    if save_all and str(state['current_epoch']) in args.save_some_models:
        shutil.copyfile(checkpoint_path, join(
            dirname,
            'checkpoint_epoch_%s.pth.tar' % state['current_epoch']))
