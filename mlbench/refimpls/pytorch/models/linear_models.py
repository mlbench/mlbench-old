import torch
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class LogisticRegression(torch.nn.Module):
    """
    Assume the data matrix is dense
    """

    def __init__(self, n_features, sparse=False):
        super(LogisticRegression, self).__init__()
        if sparse:
            raise NotImplementedError("For the moment the sparse dataset is not supported.")

        self.linear = torch.nn.Linear(n_features, 1, bias=False)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


class RidgeRegression(torch.nn.Module):
    """
    Assume the data matrix is dense
    """

    def __init__(self, n_features):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n_features, 1, bias=False)

    def forward(self, x):
        return self.linear(x)


def get_model(options):
    from datasets.load_libsvm_dataset import get_dataset_info
    info = get_dataset_info(options.dataset_name)

    if options.model_name == 'logistic_regression':
        return LogisticRegression(info['n_features'], sparse=options.sparse_dataset)
    elif options.model_name == 'ridge_regression':
        return RidgeRegression(info['n_features'])
    else:
        raise NotImplementedError("Get Model.")
