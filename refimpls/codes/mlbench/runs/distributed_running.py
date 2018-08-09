# -*- coding: utf-8 -*-
import gc
import time

import torch
import torch.distributed as dist
from torch.autograd import Variable

from mlbench.dataset.create_data import create_dataset
from mlbench.utils.log import log, log0
from mlbench.utils.meter import AverageMeter, accuracy, save_checkpoint, \
    define_local_tracker
from mlbench.utils.lr import adjust_learning_rate


def load_data(args, input, target, tracker):
    """Load a mini-batch and record the loading time."""
    # get variables.
    start_data_time = time.time()

    if args.graph.on_gpu:
        input, target = input.cuda(), target.cuda()
    input_var, target_var = Variable(input), Variable(target)

    # measure the data loading time
    end_data_time = time.time()
    tracker['data_time'].update(end_data_time - start_data_time)
    tracker['end_data_time'] = end_data_time
    return input, target, input_var, target_var


def aggregate_gradients(args, model, optimizer):
    """Aggregate gradients."""
    # all_reduce the gradients.
    for ind, param in enumerate(model.parameters()):
        # all reduce.
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)

        # if or not averge the model.
        if args.avg_model:
            param.grad.data /= args.graph.n_nodes

    # apply these gradients.
    optimizer.step(apply_lr=True)


def inference(model, criterion, input_var, target_var, target):
    """Inference on the given model and get loss and accuracy."""
    output = model(input_var)
    loss = criterion(output, target_var)
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    return loss, prec1, prec5


def train_and_validate(args, model, criterion, optimizer):
    """The training scheme of Hierarchical Local SGD."""
    # get data loader.
    train_loader, val_loader = create_dataset(args)

    # define some parameters for training.
    log0('we have {} epochs, {} mini-batches per epoch (batch size:{}).'.format(
        args.num_epochs, args.num_batches_train, args.batch_size))

    dist.barrier()
    log0('*'*80)

    # only evaluate the model if required.
    if args.evaluate:
        validate(args, val_loader, model, criterion)
        return

    # train the model and evaluate the model per args.eval_freq
    for epoch in range(args.start_epoch, args.num_epochs + 1):
        args.epoch = epoch

        # train
        do_training(args, train_loader, model, optimizer, criterion)

        # evaluate on validation set.
        if epoch % args.eval_freq == 0:
            do_validate(args, val_loader, model, optimizer, criterion)

        # reshuffle the data.
        if args.reshuffle_per_epoch:
            # TODO: A little bit too specific, should be hide beneth
            # Is it really useful to delete train loader and val loader?
            del train_loader, val_loader
            gc.collect()
            log('reshuffle the dataset.')
            train_loader, val_loader = create_dataset(args)


def do_training(args, train_loader, model, optimizer, criterion):
    # switch to train mode
    model.train()

    tracker = define_local_tracker()

    tracker['start_load_time'] = time.time()

    for iter, (input, target) in enumerate(train_loader):
        # update local step.
        tracker.logging_load(args)
        # TODO: what doe sthe local index do?
        args.local_index += 1

        # adjust learning rate (based on the # of accessed samples)
        if args.lr_decay is not None:
            adjust_learning_rate(args, optimizer)

        # load data
        input, target, input_var, target_var = load_data(
            args, input, target, tracker)

        # inference and get current performance.
        loss, prec1, prec5 = inference(
            model, criterion, input_var, target_var, target)

        # compute gradient and do local SGD step.
        optimizer.zero_grad()
        loss.backward()

        # logging locally.
        tracker.logging_computing(args, loss, prec1, prec5, input)

        # sync and apply gradients.
        aggregate_gradients(args, model, optimizer)

        # logging display.
        tracker.logging_sync(args)
        tracker.logging_display(args)

        if iter >= 5:
            break


def do_validate(args, val_loader, model, optimizer, criterion, save=True):
    """Evaluate the model on the test dataset and save to the checkpoint."""
    # evaluate the model.
    val_prec1, val_prec5 = validate(args, val_loader, model, criterion)

    # remember best prec@1 and save checkpoint.
    is_best = val_prec1 > args.best_prec1
    if is_best:
        args.best_prec1 = val_prec1
        args.best_epoch += [args.epoch]
    log('best accuracy for rank {} at lcoal index {} \
        (best epoch {}, current epoch {}): {}.'.format(
        args.graph.rank, args.local_index,
        args.best_epoch[-1] if len(args.best_epoch) != 0 else '',
        args.epoch, args.best_prec1))

    if save and args.graph.rank == 0:
        save_checkpoint(
            {
                'arguments': args,
                'current_epoch': args.epoch,
                'local_index': args.local_index,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': args.best_prec1,
            },
            is_best, dirname=args.checkpoint_root,
            filename='checkpoint.pth.tar',
            save_all=args.save_all_models)


def validate(args, val_loader, model, criterion):
    """A function for model evaluation."""
    # define stat.
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # TODO: miss a barrier here?
    # switch to evaluation mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        if args.graph.rank == 0:
            log0('Validation at batch {}/{}'.format(i, args.num_batches_val))

        # place data.
        if args.graph.on_gpu:
            input, target = input.cuda(), target.cuda()

        # inference based on the given data.
        with torch.no_grad():
            input_var = Variable(input)
            target_var = Variable(target)

            loss, prec1, prec5 = inference(
                model, criterion, input_var, target_var, target)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

    log0('Aggregate val accuracy from different partitions.')
    top1_avg, top5_avg = aggregate_accuracy(top1, top5)

    log0('Val at batch: {}. \
         Process: {}. Prec@1: {:.3f} Prec@5: {:.3f}'.format(
        args.local_index, args.graph.rank, top1_avg, top5_avg))
    return top1_avg, top5_avg


def aggregate_accuracy(top1, top5):
    def helper(array):
        array = torch.FloatTensor(array)
        dist.all_reduce(array, op=dist.reduce_op.SUM)
        return array[0] / array[1]
    top1_avg = helper([top1.sum, top1.count])
    top5_avg = helper([top5.sum, top5.count])
    return top1_avg, top5_avg
