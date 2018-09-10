import torch.optim as optim


def get_optimizer(options, model):
    """Get optimizer for the given configuration.

    Using the configurations in the `options`, create an optimizer associated with
    parameters of `model`. A learning rate for optimizer is created as well.

    :param options: A global object containing specified options.
    :type options: argparse.Namespace
    :param model: A model to be optimized by the optimizer.
    :type model: torch.nn.Module
    :returns: optimizer of the given model.
    :rtype: optimizer
    :raises: NotImplementedError
    """
    lr = options.lr if hasattr(options, 'lr') else options.lr_per_sample * options.batch_size

    if options.opt_name == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=lr,
                              momentum=options.momentum,
                              weight_decay=options.weight_decay,
                              nesterov=options.nesterov)
    else:
        raise NotImplementedError("The optimizer `{}` specified by `options` is not implemented."
                                  .format(options.opt_name))

    return optimizer
