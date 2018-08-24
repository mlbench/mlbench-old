import torch.optim as optim


def get_optimizer(options, model):
    """Get optimizer for the given configuration.

    Using the configurations in the `options`, create an optimizer associated with
    parameters of `model`. A learning rate for optimizer is created as well.

    Parameters
    ----------
    options : {argparse.Namespace}
        A options object containing all configurations.
    model : {torch.nn.Module}
        A model to be optimized by the optimizer.

    Returns
    -------
    optimizer
        optimizer of the given model.

    Raises
    ------
    NotImplementedError
        The optimizer specified by `options` is not implemented.
    """
    if options.opt_name == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=options.lr,
                              momentum=options.momentum,
                              weight_decay=options.weight_decay,
                              nesterov=options.nesterov)
    else:
        raise NotImplementedError("The optimizer `{}` specified by `options` is not implemented."
                                  .format(options.opt_name))

    return optimizer
