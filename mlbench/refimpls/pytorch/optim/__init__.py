import torch
import torch.optim as optim


from utils import log
from . import lr


def get_optimizer(context, model):
    if context.optimizer.name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=context.optimizer.lr_init,
                              momentum=context.optimizer.momentum,
                              weight_decay=context.optimizer.weight_decay, nesterov=context.optimizer.nesterov
                              )
        scheduler = lr.linear_cycle(optimizer, lr_init=context.optimizer.lr_init,
                                    epochs=context.controlflow.num_epochs, low_lr=0.005, extra=0)
    else:
        raise NotImplementedError

    return optimizer, scheduler


def get_criterion(context):
    # TODO: put criterion into a good place.
    if context.optimizer.criterion == 'CrossEntropyLoss':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

    return criterion
