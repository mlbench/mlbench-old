import time
import torch.distributed as dist

from mlbench.utils import log


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

        if debug and batch_idx >= 50:
            break

        # TODO: fine grained logging.


class Train(object):
    def __call__(self, model, optimizer, criterion, data_loader, num_epochs, num_batches, batch_size,
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

        log.centering("End training.", 0)
