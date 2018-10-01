import re
import collections
import scipy.sparse as sp
import numpy as np
import torch
from torch._six import string_classes, int_classes, FileNotFoundError

_use_shared_memory = False


def _default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: _default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [_default_collate(samples) for samples in transposed]
    elif 'scipy.sparse' in str(elem_type):
        data = sp.vstack(batch)
        i = torch.LongTensor(data.nonzero())
        v = torch.Tensor(data.data)
        shape = (len(batch), batch[0].shape[1])
        output = torch.sparse_coo_tensor(i, v, shape)
        return output

    raise TypeError((error_msg.format(type(batch[0]))))


def numpy_sparse_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    # https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader
    batch = _default_collate(batch)
    return batch
