# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['AlexNet', 'alexnet']


class AlexNet(nn.Module):
    """An AlexNet.
    It directly borrowed from
    https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
    """

    def __init__(self, args):
        super(AlexNet, self).__init__()
        num_classes = self.decide_num_classes(args)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def decide_num_classes(self, args):
        if args.data == 'cifar10':
            return 10
        elif args.data == 'cifar100':
            return 100
        elif args.data == 'imagenet':
            return 1000


def alexnet(args, pretrained=False):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = AlexNet(args)
    if pretrained:
        model.load_state_dict(
            model_zoo.load_url(
                'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
            )
        )
    return model
