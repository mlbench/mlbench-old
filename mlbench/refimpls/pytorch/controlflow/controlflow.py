import time
import torch
import torch.distributed as dist
import itertools

from utils import log
from utils import checkpoint
from utils.metrics import AverageMeter
from utils.helper import Timeit


def aggregate_gradients(model, world_size):
    """Average gradients of models across all processes.

    Parameters
    ----------
    model : {torch.nn.Module}
        a specified model for training.
    world_size : {int}
        number of processes 
    """
    # all_reduce the gradients.
    for ind, param in enumerate(model.parameters()):
        # all reduce.
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= world_size


def _counter(options):
    if options.max_batch_per_epoch is None:
        counter = itertools.count(0)
    else:
        counter = range(options.max_batch_per_epoch)
    return counter


def train_epoch(model, optimizer, criterion, options):
    """Train model for one epoch of data.

    Parameters
    ----------
    model : {torch.nn.Module}
        a specified model for training.
    optimizer : {torch.nn.Optimizer}
        an optimizer for the model.
    criterion : {torch.nn.modules.loss}
        defined loss function.
    options : {Context}
        global configurations.
    """
    # switch to train mode
    model.train()

    for batch_idx, (data, target) in zip(_counter(options), options.train_loader):
        if options.use_cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        aggregate_gradients(model, options.world_size)
        optimizer.step()

        log.train_batch(options, batch_idx, loss.item())


def validate(model, optimizer, criterion, metrics, options):
    model.eval()

    losses = AverageMeter()

    metrics.reset()
    for batch_idx, (data, target) in zip(_counter(options), options.val_loader):
        if options.use_cuda:
            data, target = data.cuda(), target.cuda()

        with torch.no_grad():
            output = model(data)

            loss = criterion(output, target)

            prec1, prec5 = metrics(output, target)

            log.debug("Validate Batch {:5}: Prec@1={:.3f} Prec@5={:.3f}"
                      .format(batch_idx, prec1.item(), prec5.item()), 0)

            losses.update(loss.item(), data.size(0))
            metrics.update(prec1, prec5, data)

    top1_avg, top5_avg = metrics.average()
    log.info('Prec@1: {:.3f} Prec@5: {:.3f} Loss: {:.3f}'.format(
        top1_avg, top5_avg, losses.avg), 0)
    return top1_avg, top5_avg


def do_validate(model, optimizer, criterion, metrics, scheduler, options, timeit):
    """Evaluate the model on the test dataset and save to the checkpoint."""
    # evaluate the model.
    val_prec1, val_prec5 = validate(model, optimizer, criterion, metrics, options)

    timeit.pause()

    # remember best prec@1 and save checkpoint.
    if 'best_prec1' in options.runtime:
        is_best = val_prec1 > options.runtime['best_prec1']
    else:
        is_best = True

    if is_best:
        options.runtime['best_prec1'] = val_prec1.item()
        options.runtime['best_epoch'] = options.runtime['current_epoch']

    log.log_val(options, val_prec1)

    checkpoint.save(options, model, optimizer, scheduler, is_best)

    timeit.resume()


class TrainValidation(object):
    def __call__(self, model, optimizer, criterion, metrics, scheduler, options):
        # define some parameters for training.
        log.info('There are {} epochs, {} mini-batches per epoch (batch size:{}).'
                 .format(options.train_epochs, options.train_num_batches,
                         options.batch_size), 0)

        # train the model and evaluate the model per args.eval_freq
        dist.barrier()
        max_epochs = min(options.train_epochs, options.max_train_steps)
        start_epoch = options.runtime['current_epoch'] if options.resume else 0

        with Timeit() as timeit:
            for epoch in range(start_epoch, max_epochs):
                options.runtime['current_epoch'] = epoch

                # schedule learning rates
                scheduler.step()

                # Per epoch information.
                log.info("Current epoch : {} : lr={} : time={:10.3e}"
                         .format(epoch, scheduler.get_lr(), timeit.cumu), 0)

                train_epoch(model, optimizer, criterion, options)
                do_validate(model, optimizer, criterion, metrics, scheduler, options, timeit)


def get_controlflow(options):
    """Get optimizer and scheduler for the given configuration.

    Using the configurations in the `options`, create an optimizer associated with
    parameters of `model`. A learning rate for optimizer is created as well.

    Parameters
    ----------
    options : {argparse.Namespace}
         A options object containing all configurations.

    Returns
    -------
    controlflow
        A controlflow object.

    Raises
    ------
    NotImplementedError
        The controlflow specified by `options` is not implemented.
    """
    return TrainValidation()
