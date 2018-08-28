import torch.distributed as dist


def elementwise_min(tensor):
    dist.all_reduce(tensor, op=dist.reduce_op.MIN)
    return tensor


def aggregate_gradients(model, world_size):
    """Average gradients of models across all processes."""
    # all_reduce the gradients.
    for ind, param in enumerate(model.parameters()):
        # all reduce.
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= world_size
