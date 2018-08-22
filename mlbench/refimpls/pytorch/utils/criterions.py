import torch


def get_criterion(options):
    try:
        criterion = 'torch.nn.modules.loss.' + options.criterion + '()'
        criterion = eval(criterion)
    except Exception as e:
        raise NotImplementedError(criterion)

    return criterion
