import argparse
import json
import config

from optim import get_optimizer, get_criterion
from controlflow import get_controlflow
from models import get_model
from datasets import get_dataset
from utils import log, get_metrics, checkpoint


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Entrypoint to distributed training of neural networks.')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--run_id', type=str, default='0', metavar='S',
                        help='Id of current run.')
    parser.add_argument('--logging-level', type=str, default='DEBUG', metavar='N',
                        help='')
    parser.add_argument('--debug', type=str2bool, default=False, metavar='N',
                        help='In debug mode some of intermediate data will be printed out.')
    parser.add_argument('--resume', type=str2bool, default=False, metavar='N',
                        help='Restore the experiment specified by run_id.')
    parser.add_argument('--config-file', type=str, default=None, metavar='N',
                        help='A json file specifying detailed information about the configuration.')
    parser.add_argument('--experiment', type=str, default=None, metavar='N',
                        help='The name of the experiment to run (added for compatibility.')
    args = parser.parse_args()

    # Initialize environements and load default settings to `context`
    context = config.init_context(args)

    # Get dataset
    get_dataset(context)

    # The model here can be any model subclass of `nn.Module`
    model = get_model(context)

    # The optimizer here can be any one subclass of `torch.optim.optimizer.Optimizer`
    optimizer, scheduler = get_optimizer(context, model)

    # Criterions are like `torch.nn.CrossEntropyLoss()`
    criterion = get_criterion(context)

    metrics = get_metrics(context)

    # Controlflow decides the main logic of training, validating or testing.
    controlflow = get_controlflow(context)

    # Resume training of `run_id` if possible
    if context.meta.resume:
        context = checkpoint.resume(context, model, optimizer, scheduler)

    # Print configuration information to terminal.
    log.configuration_information(context)

    controlflow(model=model, optimizer=optimizer, criterion=criterion,
                metrics=metrics, scheduler=scheduler, context=context)


if __name__ == '__main__':
    main()
