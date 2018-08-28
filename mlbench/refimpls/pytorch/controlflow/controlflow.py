import torch
import torch.distributed as dist

from utils import checkpoint
from utils import log
from utils.metrics import AverageMeter
from utils.helper import Timeit, maybe_range, update_best_runtime_metric
from utils.communication import aggregate_gradients


def train_epoch(model, optimizer, criterion, options):
    """Train model for one epoch of data."""
    # switch to train mode
    model.train()

    for batch_idx, (data, target) in zip(maybe_range(options.max_batch_per_epoch),
                                         options.train_loader):
        if options.use_cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        aggregate_gradients(model, options.world_size)
        optimizer.step()

        log.train_batch(options, batch_idx, loss.item())
        if options.lr_scheduler_level == 'batch':
            scheduler.step()


def validate(model, optimizer, criterion, metrics, options):
    model.eval()

    losses = AverageMeter()

    metrics.reset()
    for batch_idx, (data, target) in zip(maybe_range(options.max_batch_per_epoch),
                                         options.val_loader):
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

    is_best = update_best_runtime_metric(options, val_prec1.item(), 'prec1')

    checkpoint.save(options, model, optimizer, scheduler, is_best)

    log.log_val(options, val_prec1)

    timeit.resume()


class TrainValidation(object):
    def __call__(self, model, optimizer, criterion, metrics, scheduler, options):
        """Train models and perform validation.

        :param model: a pytorch model to be trained and validated.
        :type model: nn.Module
        :param optimizer: an optimizer for the given model.
        :param criterion: loss function. 
        :param metrics: metrics like TopKAccuracy.
        :param scheduler: a scheduler for hyperparameters.
        :param options: a global object containing all of the options.
        :type options: argparse.Namespace
        """
        # define some parameters for training.
        log.info('There are {} epochs, {} mini-batches per epoch (batch size:{}).'
                 .format(options.train_epochs, options.train_num_batches,
                         options.batch_size), 0)

        # train the model and evaluate the model per args.eval_freq
        max_epochs = min(options.train_epochs, options.max_train_steps)\
            if options.max_train_steps else options.train_epochs
        start_epoch = options.runtime['current_epoch'] if options.resume else 0

        dist.barrier()

        timeit = Timeit()
        for epoch in range(start_epoch, max_epochs):
            options.runtime['current_epoch'] = epoch

            # schedule learning rates
            if options.lr_scheduler_level == 'epoch':
                scheduler.step()

            # Per epoch information.
            log.info("Current epoch : {} : lr={} : time={:10.3e}"
                     .format(epoch, scheduler.get_lr(), timeit.cumu), 0)

            train_epoch(model, optimizer, criterion, options)
            do_validate(model, optimizer, criterion, metrics, scheduler, options, timeit)


def get_controlflow(options):
    return TrainValidation()
