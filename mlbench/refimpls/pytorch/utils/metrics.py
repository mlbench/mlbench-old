import torch
import torch.distributed as dist

from utils.communication import global_average


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


class TopKAccuracy(object):
    def __init__(self, topk=1):
        self.topk = topk
        self.reset()

    def __call__(self, output, target):
        """Computes the precision@k for the specified values of k"""
        batch_size = target.size(0)

        _, pred = output.topk(self.topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:self.topk].view(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / batch_size)

    def reset(self):
        self.top = AverageMeter()

    def update(self, prec, size):
        self.top.update(prec, size)

    def average(self):
        return global_average(self.top.sum, self.top.count)

    @property
    def name(self):
        return "Prec@{}".format(self.topk)


def get_metrics(options):
    if options.metrics == 'topk':
        return [TopKAccuracy(topk=1), TopKAccuracy(topk=5)]
    if options.metrics == 'none':
        return []
    else:
        raise NotImplementedError('No metrics name {} found.'.format(options.metrics))
