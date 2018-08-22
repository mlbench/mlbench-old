import torch
import torch.distributed as dist


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


def aggregate_accuracy(top1, top5):
    def helper(array):
        array = torch.FloatTensor(array)
        dist.all_reduce(array, op=dist.reduce_op.SUM)
        return array[0] / array[1]
    top1_avg = helper([top1.sum, top1.count])
    top5_avg = helper([top5.sum, top5.count])
    return top1_avg, top5_avg


class TopKAccuracy(object):
    def __init__(self, topk=(1,)):
        self.topk = topk
        self.reset()

    def __call__(self, output, target):
        """Computes the precision@k for the specified values of k"""
        topk = self.topk

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

    def reset(self):
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()

    def update(self, prec1, prec5, data):
        self.top1.update(prec1[0], data.size(0))
        self.top5.update(prec5[0], data.size(0))

    def average(self):
        return aggregate_accuracy(self.top1, self.top5)


def get_metrics(options):
    return TopKAccuracy(topk=(1, 5))
