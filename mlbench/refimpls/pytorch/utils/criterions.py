import torch


def get_criterion(options):
    try:
        pytorch_criterion_class = getattr(torch.nn.modules.loss, options.criterion)
        criterion = pytorch_criterion_class()
    except Exception as e:
        raise NotImplementedError(criterion)

    return criterion
