import torch
import torch.distributed as dist

from utils import checkpoint
from utils import log
from utils.metrics import AverageMeter
from utils.helper import Timeit, maybe_range, update_best_runtime_metric
from utils.communication import aggregate_gradients, global_average

from utils.utils import convert_dtype


def train_epoch(model, optimizer, criterion, scheduler, options, timeit):
    """Train model for one epoch of data."""

    # switch to train mode
    model.train()

    for batch_idx, (data, target) in zip(maybe_range(options.max_batch_per_epoch),
                                         options.train_loader):
        if options.lr_scheduler_level == 'batch':
            scheduler.step()

        data = convert_dtype(options, data)
        if options.force_target_dtype:
            target = convert_dtype(options, target)

        if options.use_cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        aggregate_gradients(model, options.world_size)
        optimizer.step()

        with torch.no_grad():
            loss = loss.item()
            loss = global_average(loss, 1).item()
            log.debug("Train Batch {:5}: loss={:.3f}".format(batch_idx, loss), 0)
            log.post_metrics(options, 'train_loss', loss)

            timeit.pause()
            options.runtime['cumu_time_train'].append(timeit.cumu)
            timeit.resume()


def validate(model, optimizer, criterion, metrics, options):
    model.eval()

    losses = AverageMeter()
    for metric in metrics:
        metric.reset()

    for batch_idx, (data, target) in zip(maybe_range(options.max_batch_per_epoch),
                                         options.val_loader):
        data = convert_dtype(options, data)
        if options.force_target_dtype:
            target = convert_dtype(options, target)

        if options.use_cuda:
            data, target = data.cuda(), target.cuda()

        with torch.no_grad():
            output = model(data)

            loss = criterion(output, target)
            losses.update(loss.item(), data.size(0))

            for metric in metrics:
                metric_value = metric(output, target)
                metric.update(metric_value, data.size(0))

    metrics_averages = {metric.name: metric.average().item() for metric in metrics}
    loss_average = global_average(losses.sum, losses.count).item()
    return metrics_averages, loss_average


def do_validate(model, optimizer, criterion, metrics, scheduler, options, timeit):
    """Evaluate the model on the test dataset and save to the checkpoint."""
    # evaluate the model.
    metrics_values, loss = validate(model, optimizer, criterion, metrics, options)

    options.runtime['cumu_time_val'].append(timeit.cumu)

    if len(metrics_values) > 0:
        # Assume the first metric is used to determine the best model to checkpoint.
        prim_metric = metrics[0]
        prim_metric_value = metrics_values[prim_metric.name]

        is_best, best_metric_name = update_best_runtime_metric(options, prim_metric_value, prim_metric.name)

        log.log_val(options, best_metric_name)

        for name, value in metrics_values.items():
            log.post_metrics(options, name, value)
    else:
        is_best = False
        log.debug("Validation loss={:.3f}".format(loss), 0)

    log.post_metrics(options, 'val_loss', loss)

    checkpoint.save(options, model, optimizer, scheduler, is_best)


class Train(object):
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
        options.runtime['records'] = options.runtime.get('records', [])
        options.runtime['cumu_time_val'] = options.runtime.get('cumu_time_val', [])
        options.runtime['cumu_time_train'] = options.runtime.get('cumu_time_train', [])

        dist.barrier()

        timeit = Timeit(0 if len(options.runtime['cumu_time_val']) == 0
                        else options.runtime['cumu_time_val'][-1])
        for epoch in range(start_epoch, max_epochs):
            options.runtime['current_epoch'] = epoch

            # schedule learning rates
            if options.lr_scheduler_level == 'epoch':
                scheduler.step()

            # Per epoch information.
            log.info("Current epoch : {} : lr={} : time={:10.3e}"
                     .format(epoch, scheduler.get_lr(), timeit.cumu), 0)

            train_epoch(model, optimizer, criterion, scheduler, options, timeit)

            if not options.no_validation:
                timeit.pause()
                do_validate(model, optimizer, criterion, metrics, scheduler, options, timeit)
                timeit.resume()


def get_controlflow(options):
    return Train()
