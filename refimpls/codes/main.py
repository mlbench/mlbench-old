import argparse
import json
import config

from mlbench.optim import get_optimizer
from mlbench.controlflow import get_controlflow
from mlbench.models import get_model
from mlbench.datasets import get_dataset


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='mlbench.')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--config-file', type=str, default=None, metavar='N',
                        help='A json file specifying detailed information about the configuration.')
    args = parser.parse_args()

    # Initialize environements like
    context = config.init_context(args)

    # The arugment passed from command line has higher priority than config files.
    model = get_model(context.model)

    # Get optimizer
    optimizer = get_optimizer(context.optimizer)

    # Get control flow
    controlflow = get_controlflow(context.controlflow)

    # Get dataset
    dataset = get_dataset(context.dataset)

    # Real execution
    controlflow(model, optimizer, dataset, context)


if __name__ == '__main__':
    main()
