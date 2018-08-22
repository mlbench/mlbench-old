import torch
import torch.optim as optim


from utils import log
from . import lr


def get_optimizer(context, model):
    """Get optimizer and scheduler for the given configuration.

    Using the configurations in the `context`, create an optimizer associated with
    parameters of `model`. A learning rate for optimizer is created as well.

    Parameters
    ----------
    context : {Context}
        A context object containing all configurations.
    model : {torch.nn.Module}
        A model to be optimized by the optimizer.

    Returns
    -------
    optimizer, scheduler
        optimizer and scheduler of the given model.

    Raises
    ------
    NotImplementedError
        The optimizer specified by `context` is not implemented.
    """
    if context.optimizer.name == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=context.optimizer.lr_init,
                              momentum=context.optimizer.momentum,
                              weight_decay=context.optimizer.weight_decay,
                              nesterov=context.optimizer.nesterov)
        scheduler = lr.linear_cycle(optimizer, lr_init=context.optimizer.lr_init,
                                    epochs=context.controlflow.num_epochs, low_lr=0.005, extra=0)
    else:
        raise NotImplementedError("The optimizer `{}` specified by `context` is not implemented."
                                  .format(context.optimizer.name))

    return optimizer, scheduler


def get_criterion(context):
    """Get the criterion for training neural network.

    Parameters
    ----------
    context : {Context}
        A context object containing all configurations.

    Returns
    -------
    criterion : {torch.nn.modules.loss}
        A loss function for training.

    Raises
    ------
    NotImplementedError
        The criterion specified by `context` is not implemented.
    """
    if context.optimizer.criterion == 'CrossEntropyLoss':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError("The criterion `{}` specified by `context` is not implemented."
                                  .format(context.optimizer.criterion))

    return criterion
