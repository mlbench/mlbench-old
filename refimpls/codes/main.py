import argparse
import json
import config

from mlbench.optim import get_optimizer, get_criterion
from mlbench.controlflow import get_controlflow
from mlbench.models import get_model
from mlbench.datasets import get_dataset
from mlbench.utils import log

from mlbench import distributed_running


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='mlbench.')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--logging-level', type=str, default='DEBUG', metavar='N',
                        help='')
    parser.add_argument('--debug', type=bool, default=True, metavar='N',
                        help='In debug mode some of intermediate data will be printed out.')
    parser.add_argument('--config-file', type=str, default=None, metavar='N',
                        help='A json file specifying detailed information about the configuration.')
    args = parser.parse_args()

    # Initialize environements like
    context = config.init_context(args)

    # Get dataset
    get_dataset(context)

    # The arugment passed from command line has higher priority than config files.
    model = get_model(context)

    # Get optimizer
    optimizer = get_optimizer(context, model)

    criterion = get_criterion(context)

    # Get control flow
    controlflow = get_controlflow(context)

    # Real execution
    distributed_running(controlflow, model, optimizer, criterion, context)


if __name__ == '__main__':
    main()
