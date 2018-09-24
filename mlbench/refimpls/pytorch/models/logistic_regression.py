import torch
import torch.nn.functional as F


class LogisticRegression(torch.nn.Module):
    def __init__(self, n_features):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n_features, 1, bias=False)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


def get_model(options):
    from datasets.load_libsvm_dataset import get_dataset_info
    info = get_dataset_info(options.dataset_name)
    return LogisticRegression(info['n_features'])
