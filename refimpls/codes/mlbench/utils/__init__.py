from .metrics import Accuracy


def get_metrics(context):
    if context.meta.metrics == 'accuracy':
        metric = Accuracy(context.meta.topk)
    else:
        raise NotImplementedError
    return metric
