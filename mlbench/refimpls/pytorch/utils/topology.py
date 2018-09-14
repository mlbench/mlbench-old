import socket
import torch
import torch.distributed as dist


def _ranks_on_same_node(rank, world_size):
    hostname = socket.gethostname()
    hostname_length = torch.IntTensor([len(hostname)])
    dist.all_reduce(hostname_length, op=dist.reduce_op.MAX)
    max_hostname_length = hostname_length.item()

    encoding = [ord(c) for c in hostname]
    encoding += [-1 for c in range(max_hostname_length - len(hostname))]
    encoding = torch.IntTensor(encoding)

    all_encodings = [torch.IntTensor([0] * max_hostname_length) for _ in range(world_size)]
    dist.all_gather(all_encodings, encoding)

    all_encodings = [ec.numpy().tolist() for ec in all_encodings]
    counter = 0
    for i in range(rank):
        if all_encodings[rank] == all_encodings[i]:
            counter += 1
    return counter


class FCGraph(object):
    def __init__(self, options):
        self.rank = options.rank
        self.world_size = options.world_size
        self.use_cuda = options.use_cuda

    @property
    def current_device_name(self):
        return 'cuda:{}'.format(torch.cuda.current_device())\
            if self.use_cuda else 'cpu'

    @property
    def current_device(self):
        return torch.device(self.current_device_name())

    def assigned_gpu_id(self):
        num_gpus_on_device = torch.cuda.device_count()
        assigned_id = _ranks_on_same_node(self.rank, self.world_size)
        torch.cuda.set_device(assigned_id)

    def __str__(self):
        return "{}".format(self.current_device_name)

    def __repr__(self):
        return self.__str__()
