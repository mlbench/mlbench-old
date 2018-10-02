import unittest
import torch.optim
import torchvision.models as models
import torch.nn as nn
import copy

from argparse import Namespace
from optim.lr import multistep_learning_rates_with_warmup


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class TestMultistepLearningRatesWithWarmup(unittest.TestCase):

    def test_scaling(self):
        options = Namespace(world_size=11, warmup_linear_scaling=True, warmup_init_lr_nonscale=True,
                            lr_per_sample=0.1, batch_size=32, lr=1000000, lr_scheduler_level='epoch',
                            multisteplr_gamma=0.1, warmup=True, warmup_init_lr=0.0,
                            warmup_durations={'epoch': 5}, multisteplr_milestones={'epoch': [82, 109]})

        optimizer = torch.optim.SGD(Net().parameters(), lr=99999, momentum=0.9, weight_decay=0.0001, nesterov=True)

        scheduler = multistep_learning_rates_with_warmup(options, optimizer)

        expected_base_lr = options.world_size * options.lr_per_sample * options.batch_size
        self.assertEqual(scheduler.base_lrs, [expected_base_lr])

        expected_init_lr = options.lr_per_sample * options.batch_size

        # The first epoch is
        scheduler.step()
        self.assertAlmostEqual(scheduler.get_lr()[0], expected_init_lr)

        for _ in range(1, options.warmup_durations['epoch'] + 1):
            scheduler.step()
        self.assertAlmostEqual(scheduler.get_lr()[0], expected_base_lr)

        for _ in range(options.warmup_durations['epoch'] + 1, options.multisteplr_milestones['epoch'][0]):
            scheduler.step()
        self.assertAlmostEqual(scheduler.get_lr()[0], expected_base_lr)

        scheduler.step()
        self.assertAlmostEqual(scheduler.get_lr()[0], expected_base_lr * options.multisteplr_gamma)

        for _ in range(options.multisteplr_milestones['epoch'][0] + 1, options.multisteplr_milestones['epoch'][1]):
            scheduler.step()
        self.assertAlmostEqual(scheduler.get_lr()[0], expected_base_lr * options.multisteplr_gamma)

        scheduler.step()
        self.assertAlmostEqual(scheduler.get_lr()[0], expected_base_lr * options.multisteplr_gamma ** 2)


if __name__ == '__main__':
    unittest.main()
