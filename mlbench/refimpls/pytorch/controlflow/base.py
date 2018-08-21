import time
import torch
import torch.distributed as dist
import datetime

from utils import log
from utils import checkpoint
from utils.metrics import AverageMeter


def aggregate_gradients(model, world_size):
    # all_reduce the gradients.
    for ind, param in enumerate(model.parameters()):
        # all reduce.
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= world_size


def train_epoch(model, optimizer, criterion, context):
    # switch to train mode
    model.train()

    for batch_idx, (data, target) in enumerate(context.dataset.train_.loader):
        if context.meta.use_cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        aggregate_gradients(model, context.meta.world_size)
        optimizer.step()

        log.debug("Training Batch {:5}: loss={:.3f}".format(batch_idx, loss.item()))

        log.post_metrics({
            "run_id": context.meta.run_id,
            "name": "train loss @ rank{}".format(context.meta.rank),
            "value": loss.item(),
            "date": str(datetime.datetime.now()),
            "cumulative": False,
            "metadata":
            "Training loss at rank {}, epoch {} and batch {}".format(
                context.meta.rank, context.runtime.current_epoch,
                batch_idx
            )
        }, context.meta.rank)
        if context.meta.debug and batch_idx >= 10:
            break


def aggregate_accuracy(top1, top5):
    def helper(array):
        array = torch.FloatTensor(array)
        dist.all_reduce(array, op=dist.reduce_op.SUM)
        return array[0] / array[1]
    top1_avg = helper([top1.sum, top1.count])
    top5_avg = helper([top5.sum, top5.count])
    return top1_avg, top5_avg


def validate(model, optimizer, criterion, metrics, context):
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for batch_idx, (data, target) in enumerate(context.dataset.val_.loader):
        if context.meta.use_cuda:
            data, target = data.cuda(), target.cuda()

        with torch.no_grad():
            output = model(data)

            loss = criterion(output, target)
            losses.update(loss.item(), data.size(0))

            prec1, prec5 = metrics(output, target)

            log.debug("Validate Batch {:5}: Prec@1={:.3f} Prec@5={:.3f}"
                      .format(batch_idx, prec1.item(), prec5.item()), 0)

            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))

        if context.meta.debug and batch_idx >= 10:
            break

    top1_avg, top5_avg = aggregate_accuracy(top1, top5)
    log.info('Prec@1: {:.3f} Prec@5: {:.3f}'.format(top1_avg, top5_avg), 0)
    return top1_avg, top5_avg


def do_validate(model, optimizer, criterion, metrics, context):
    """Evaluate the model on the test dataset and save to the checkpoint."""
    # evaluate the model.
    val_prec1, val_prec5 = validate(model, optimizer, criterion, metrics, context)

    # remember best prec@1 and save checkpoint.
    if hasattr(context.runtime, 'best_prec1'):
        is_best = val_prec1 > context.runtime.best_prec1
    else:
        is_best = True

    if is_best:
        context.runtime.best_prec1 = val_prec1
        context.runtime.best_epoch += [context.runtime.current_epoch]

    log.info('best accuracy for rank {}:(best epoch {}, current epoch {}): {:.3f} %'.format(
        context.meta.rank,
        context.runtime.best_epoch[-1] if len(context.runtime.best_epoch) != 0 else '',
        context.runtime.current_epoch, context.runtime.best_prec1), 0)

    if context.meta.save:
        checkpoint.save({
            'context': context,
            'current_epoch': context.runtime.current_epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_prec1': context.runtime.best_prec1,
        }, is_best, context)

    log.post_metrics({
        "run_id": context.meta.run_id,
        "name": "Prec@1",
        "value": "{:.3f}".format(val_prec1),
        "date": str(datetime.datetime.now()),
        "cumulative": False,
        "metadata":
        "Validation Prec1 at epoch {}".format(
            context.runtime.current_epoch
        )
    }, context.meta.rank)


class TrainValidation(object):
    def __call__(self, model, optimizer, criterion, metrics, scheduler, context):
        # define some parameters for training.
        log.info('There are {} epochs, {} mini-batches per epoch (batch size:{}).'
                 .format(context.controlflow.num_epochs,
                         context.dataset.train_.num_batches,
                         context.dataset.batch_size), 0)
        dist.barrier()

        # train the model and evaluate the model per args.eval_freq
        for epoch in range(context.controlflow.start_epoch, context.controlflow.num_epochs):
            context.runtime.current_epoch = epoch

            # schedule learning rates
            scheduler.step()

            log.info("Current epoch : {} : lr={}".format(epoch, scheduler.get_lr()), 0)

            train_epoch(model, optimizer, criterion, context)
            do_validate(model, optimizer, criterion, metrics, context)
