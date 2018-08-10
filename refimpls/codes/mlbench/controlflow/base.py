import time
import torch
import torch.distributed as dist

from mlbench.utils import log
from mlbench.utils.metrics import AverageMeter


def average_model_weights(model, world_size):
    # all_reduce the gradients.
    for ind, param in enumerate(model.parameters()):
        # all reduce.
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= world_size


def train_epoch(data_loader, model, optimizer, criterion, use_cuda, avg_model,
                world_size, debug):
    log.warning("Check how the gradient step will influence the backprop.", 0)

    # switch to train mode
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        average_model_weights(model, world_size)
        optimizer.step()

        log.debug("Training Batch {}: loss={:.3f}".format(batch_idx, loss.item()))
        if debug and batch_idx >= 10:
            break

        # TODO: fine grained logging.


def aggregate_accuracy(top1, top5):
    def helper(array):
        array = torch.FloatTensor(array)
        dist.all_reduce(array, op=dist.reduce_op.SUM)
        return array[0] / array[1]
    top1_avg = helper([top1.sum, top1.count])
    top5_avg = helper([top5.sum, top5.count])
    return top1_avg, top5_avg


def validate(data_loader, model, optimizer, criterion, metrics, use_cuda, world_size, debug):
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for batch_idx, (data, target) in enumerate(data_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)
            prec1, prec5 = metrics(output, target)

            log.debug("Validation Batch {}: Prec@1={:.3f} Prec@5={:.3f}"
                      .format(batch_idx, prec1.item(), prec5.item()))

            losses.update(loss.item(), data.size(0))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))

        if debug and batch_idx >= 10:
            break

    top1_avg, top5_avg = aggregate_accuracy(top1, top5)
    return top1_avg, top5_avg


class TrainValidation(object):
    def __call__(self, model, optimizer, criterion, metrics, data_loader, num_epochs, num_batches, batch_size,
                 start_epoch, use_cuda, avg_model, world_size, debug):
        log.centering("Begin training.", 0)

        # define some parameters for training.
        log.info('There are {} epochs, {} mini-batches per epoch (batch size:{}).'
                 .format(num_epochs, num_batches, batch_size), 0)

        log.todo("TODO: the inference we used here implicitly assumes classification problem.", 0)

        dist.barrier()

        # train the model and evaluate the model per args.eval_freq
        log.todo("TODO: Save the current epoch id to context.", 0)
        for epoch in range(start_epoch, num_epochs + 1):
            log.debug("Begin epoch {}".format(epoch), 0)
            train_epoch(data_loader, model, optimizer, criterion, use_cuda, avg_model, world_size, debug)

            top1_avg, top5_avg = validate(data_loader, model, optimizer, criterion,
                                          metrics, use_cuda, world_size, debug)
            log.info('Prec@1: {:.3f} Prec@5: {:.3f}'.format(top1_avg, top5_avg), 0)

        log.centering("End training.", 0)
