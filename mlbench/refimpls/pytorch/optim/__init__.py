import torch
import torch.optim as optim


from utils import log


def set_lr(context):
    # args.lr_change_epochs = [
    #     int(l) for l in args.lr_decay_epochs.split(',')] \
    #     if args.lr_decay_epochs is not None \
    #     else None
    # args.learning_rate_per_sample = 0.1 / args.base_batch_size
    # args.learning_rate = \
    #     args.learning_rate_per_sample * args.batch_size * args.graph.n_nodes \
    #     if args.lr_scale else args.lr
    # args.old_learning_rate = args.learning_rate
    log.todo("TODO: schedule learning rate.", 0)


def get_optimizer(context, model):
    if context.optimizer.name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=context.optimizer.lr,
                              momentum=context.optimizer.momentum)
    else:
        raise NotImplementedError

    set_lr(context)

    return optimizer


def get_criterion(context):
    # TODO: put criterion into a good place.
    if context.optimizer.criterion == 'CrossEntropyLoss':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

    return criterion
