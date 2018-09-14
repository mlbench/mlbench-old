import torch


class FCGraph(object):
    def __init__(self, meta):
        self.rank = meta.rank
        self.world_size = meta.world_size
        self.use_cuda = meta.use_cuda

    @property
    def current_device_name(self):
        return 'cuda:{}'.format(torch.cuda.current_device())\
            if self.use_cuda else 'cpu'

    @property
    def current_device(self):
        return torch.device(self.current_device_name())

    def assigned_gpu_id(self):
        num_gpus_on_device = torch.cuda.device_count()
        assigned_id = self.rank % num_gpus_on_device
        torch.cuda.set_device(assigned_id)

    def __str__(self):
        return "{}".format(self.current_device_name)

    def __repr__(self):
        return self.__str__()
