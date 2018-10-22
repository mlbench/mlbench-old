import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELossRegularized(nn.modules.loss._WeightedLoss):
    """
    Add l1/l2 regularized terms to Binary Cross Entropy (BCE).
    """

    def __init__(self, weight=None, size_average=None, reduce=None, l1=0.0, l2=0.0, model=None,
                 reduction='elementwise_mean'):
        super(BCELossRegularized, self).__init__(weight, size_average, reduce, reduction)
        self.l2 = l2
        self.l1 = l1
        self.model = model

    def forward(self, input, target):
        output = F.binary_cross_entropy(input, target, weight=self.weight, reduction=self.reduction)
        l2_loss = sum(param.norm(2) ** 2 for param in self.model.parameters())
        output += self.l2 / 2 * l2_loss
        l1_loss = sum(param.norm(1) for param in self.model.parameters())
        output += self.l1 * l1_loss
        return output


class MSELossRegularized(nn.modules.loss._WeightedLoss):
    """
    Add l1/l2 regularized terms to Mean Squared Error (MSE).
    """

    def __init__(self, weight=None, size_average=None, reduce=None, l1=0.0, l2=0.0, model=None,
                 reduction='elementwise_mean'):
        super(MSELossRegularized, self).__init__(weight, size_average, reduce, reduction)
        self.l2 = l2
        self.l1 = l1
        self.model = model

    def forward(self, input, target):
        output = F.mse_loss(input, target, reduction=self.reduction)
        l2_loss = sum(param.norm(2) ** 2 for param in self.model.parameters())
        output += self.l2 / 2 * l2_loss
        l1_loss = sum(param.norm(1) for param in self.model.parameters())
        output += self.l1 * l1_loss
        return output


def get_criterion(options, model):
    if options.criterion == 'BCELossRegularized':
        criterion = BCELossRegularized(l1=options.l1_coef, l2=options.l2_coef, model=model)
    elif options.criterion == 'MSELossRegularized':
        criterion = MSELossRegularized(l1=options.l1_coef, l2=options.l2_coef, model=model)
    else:
        try:
            pytorch_criterion_class = getattr(torch.nn.modules.loss, options.criterion)
            criterion = pytorch_criterion_class()
        except Exception as e:
            raise NotImplementedError(criterion)

    return criterion
