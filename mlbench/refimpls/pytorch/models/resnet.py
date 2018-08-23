import torch
import torch.nn as nn
from torch.nn import functional as F


class PreActBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(PreActBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


class ResNet18_CIFAR10(nn.Module):

    def __init__(self, layers, num_classes=1000):
        super(ResNet18_CIFAR10, self).__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layers = nn.Sequential(
            self._make_layer(64, 64, layers[0], stride=1),
            self._make_layer(64, 128, layers[1], stride=2),
            self._make_layer(128, 256, layers[2], stride=2),
            self._make_layer(256, 256, layers[3], stride=2),
        )

        self.classifier = nn.Linear(512, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):

        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(PreActBlock(in_channels=in_channels, out_channels=out_channels, stride=stride))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.prep(x)

        x = self.layers(x)

        x_avg = F.adaptive_avg_pool2d(x, (1, 1))
        x_avg = x_avg.view(x_avg.size(0), -1)

        x_max = F.adaptive_max_pool2d(x, (1, 1))
        x_max = x_max.view(x_max.size(0), -1)

        x = torch.cat([x_avg, x_max], dim=-1)

        x = self.classifier(x)

        return x


def resnet18_bkj(options):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet18_CIFAR10([2, 2, 2, 2], num_classes=options.num_classes)
    return model


def get_model(options):
    if options.model_name == 'resnet18':
        if options.model_version == 'default':
            return resnet18_bkj(options)

    raise NotImplementedError("{}_{} is not implemented.".format(
        options.model_name, options.model_version))
