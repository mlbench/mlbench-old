import torch
import torch.distributed as dist
import timeit


def elementwise_min(tensor):
    dist.all_reduce(tensor, op=dist.reduce_op.MIN)
    return tensor


def aggregate_gradients(model, world_size):
    """Average gradients of models across all processes."""
    # all_reduce the gradients.
    for ind, param in enumerate(model.parameters()):
        # all reduce.
        # start = timeit.default_timer()
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        # end = timeit.default_timer() - start
        # with open(str(dist.get_rank()) + "_communication_time.txt", "a+") as file:
        #     file.write(str(end) + "\n")

        param.grad.data /= world_size


def aggregate_sparsified_gradients(model, world_size, sparse_vector_size, random_sparse, optimizer, lr):
    """Make the gradients vector sparse and average sparsified gradients of models across all processes."""

    params_sparse_tensors = optimizer.sparsify_gradients(model, lr, random_sparse)

    for ind, param in enumerate(model.parameters()):

        gathered_list = [torch.zeros_like(params_sparse_tensors[ind]) for _ in range(world_size)]
        # all gather.
        # start = timeit.default_timer()
        dist.all_gather(gathered_list, params_sparse_tensors[ind])
        # end = timeit.default_timer() - start
        # with open(str(dist.get_rank()) + "_communication_time.txt", "a+") as file:
        #     file.write(str(end) + "\n")

        avg_grads = torch.zeros_like(param.data)

        if random_sparse:
            for grad_tensor in gathered_list:
                for index in range(grad_tensor.size()[1]):
                    avg_grads[0, int(grad_tensor[0, index])] += grad_tensor[1, index]
        else:
            for grad_tensor in gathered_list:
                begin = int(grad_tensor[0, sparse_vector_size])
                avg_grads[0, begin:begin + sparse_vector_size] += grad_tensor[0, 0:sparse_vector_size]

        avg_grads /= world_size
        param.grad.data = avg_grads


def global_average(sum, count):
    def helper(array):
        array = torch.FloatTensor(array)
        dist.all_reduce(array, op=dist.reduce_op.SUM)
        return array[0] / array[1]

    avg = helper([sum, count])
    return avg
