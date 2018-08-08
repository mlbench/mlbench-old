# -*- coding: utf-8 -*-
import shutil
from os.path import join
import torch
import time
from mlbench.utils.log import log, log0


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


class Tracker(dict):
    # TODO: check the content.
    def __init__(self, *args, **kwargs):
        super(Tracker, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def logging_computing(self, args, loss, prec1, prec5, input):
        # measure accuracy and record loss.
        self['losses'].update(loss.item(), input.size(0))
        self['top1'].update(prec1[0], input.size(0))
        self['top5'].update(prec5[0], input.size(0))

        # measure elapsed time.
        self['batch_time'].update(time.time() - self['end_data_time'])
        self['start_sync_time'] = time.time()

    def logging_sync(self, args):
        # measure elapsed time.
        self['sync_time'].update(time.time() - self['start_sync_time'])

    def logging_load(self, args):
        # measure elapsed time.
        self['load_time'].update(time.time() - self['start_load_time'])

    def logging_display(self, args):
        log_info = ('Local index: {local_index}. Load: {load:.3f}s | Data: {data:.3f}s |'
                    ' Batch: {batch:.3f}s | Sync: {sync:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} |'
                    ' top5: {top5: .4f}').format(
            local_index=args.local_index,
            load=self['load_time'].avg,
            data=self['data_time'].avg,
            batch=self['batch_time'].avg,
            sync=self['sync_time'].avg,
            loss=self['losses'].avg,
            top1=self['top1'].avg,
            top5=self['top5'].avg)
        log('Process {}: '.format(args.graph.rank) + log_info)
        self['start_load_time'] = time.time()


def define_local_tracker():
    # TODO: Explain the meaning
    batch_time = AverageMeter()
    data_time = AverageMeter()
    sync_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    load_time = AverageMeter()
    tracker = Tracker({
        'batch_time': batch_time,
        'data_time': data_time,
        'sync_time': sync_time,
        'load_time': load_time,
        'losses': losses,
        'top1': top1,
        'top5': top5,
        'others': {}})
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
