from .metrics import TopKAccuracy


def get_metrics(context):
    if context.meta.metrics == 'accuracy':
        metric = TopKAccuracy(context.meta.topk)
    else:
        raise NotImplementedError
    return metric
