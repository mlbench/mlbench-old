import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer, required
import numpy as np


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
    lr = options.lr if options.lr else options.lr_per_sample * options.batch_size

    if options.opt_name == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=lr,
                              momentum=options.momentum,
                              weight_decay=options.weight_decay,
                              nesterov=options.nesterov)
    elif options.opt_name == 'sparsified_sgd':
        optimizer = sparsified_SGD(model.parameters(),
                                   lr=lr,
                                   weight_decay=options.weight_decay,
                                   sparse_grad_size=options.sparse_grad_size)
    else:
        raise NotImplementedError("The optimizer `{}` specified by `options` is not implemented."
                                  .format(options.opt_name))

    return optimizer


class sparsified_SGD(Optimizer):
    r"""Implements sparsified version of stochastic gradient descent.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        sparse_grad_size (int): Size of the sparsified gradients vector.

    """

    def __init__(self, params, lr=required, weight_decay=0, sparse_grad_size=10):

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay)

        super(sparsified_SGD, self).__init__(params, defaults)

        self.__create_gradients_memory()
        self.__create_weighted_average_params()

        self.num_coordinates = sparse_grad_size

    def __setstate__(self, state):
        super(sparsified_SGD, self).__setstate__(state)

    def __create_weighted_average_params(self):
        """ Create a memory to keep the weighted average of parameters in each iteration """
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['estimated_w'] = torch.zeros_like(p.data)
                p.data.normal_(0, 0.01)
                param_state['estimated_w'].copy_(p.data)

    def __create_gradients_memory(self):
        """ Create a memory to keep gradients that are not used in each iteration """
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['memory'] = torch.zeros_like(p.data)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            weight_decay = group['weight_decay']

            for p in group['params']:

                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                p.data.add_(-d_p)

        return loss

    def sparsify_gradients(self, model, lr, random_sparse):
        """ Calls one of the sparsification functions (random or blockwise)"""
        if random_sparse:
            return self._random_sparsify(model, lr)
        else:
            return self._block_sparsify(model, lr)

    def _random_sparsify(self, model, lr):
        """
        Sparsify the gradients vector by selecting 'k' of them randomly.
        param model: learning model
        param lr: learning rate
        return: sparsified gradients vector ('k' gradients and their indices)
        """
        params_sparse_tensors = []

        for ind, param in enumerate(model.parameters()):
            self.state[param]['memory'] += param.grad.data * lr[0]

            indices = np.random.choice(param.data.size()[1], self.num_coordinates, replace=False)
            sparse_tensor = torch.zeros(2, self.num_coordinates)

            for i, random_index in enumerate(indices):
                sparse_tensor[1, i] = self.state[param]['memory'][0, random_index]
                self.state[param]['memory'][0, random_index] = 0
            sparse_tensor[0, :] = torch.tensor(indices)

            params_sparse_tensors.append(sparse_tensor)

        return params_sparse_tensors

    def _block_sparsify(self, model, lr):
        """
        Sparsify the gradients vector by selecting a block of them.
        param model: learning model
        param lr: learning rate
        return: sparsified gradients vector
        """
        params_sparse_tensors = []

        for param in model.parameters():
            self.state[param]['memory'] += param.grad.data * lr[0]

            num_block = int(param.data.size()[1] / self.num_coordinates)

            self.current_block = np.random.randint(0, num_block)
            begin_index = self.current_block * self.num_coordinates

            if self.current_block == (num_block - 1):
                end_index = param.data.size()[1] - 1
            else:
                end_index = begin_index + self.num_coordinates - 1
            output_size = 1 + end_index - begin_index + 1

            sparse_tensor = torch.zeros(1, output_size)
            sparse_tensor[0, 0] = begin_index
            sparse_tensor[0, 1:] = self.state[param]['memory'][0, begin_index: end_index + 1]
            self.state[param]['memory'][0, begin_index: end_index + 1] = 0

            self.current_block += 1
            params_sparse_tensors.append(sparse_tensor)
        return params_sparse_tensors

    def update_estimated_weights(self, model, iteration, sparse_vector_size):
        """ Updates the estimated parameters """
        t = iteration
        for ind, param in enumerate(model.parameters()):
            tau = param.data.size()[1] / sparse_vector_size
            rho = 6 * ((t + tau) ** 2) / ((1 + t) * (6 * (tau ** 2) + t + 6 * tau * t + 2 * (t ** 2)))
            self.state[param]['estimated_w'] = self.state[param]['estimated_w'] * (1 - rho) + param.data * rho

    def get_estimated_weights(self, model):
        """ Returns the weighted average parameter tensor """
        estimated_params = []
        for param in model.parameters():
            estimated_params.append(self.state[param]['estimated_w'])
        return estimated_params
