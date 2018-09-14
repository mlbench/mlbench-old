import argparse
import re

from config import initialize
from datasets.load_dataset import create_dataset
from utils.parser import MainParser
from models import get_model
from optim.lr import get_scheduler
from optim.optimizer import get_optimizer
from controlflow.controlflow import get_controlflow
from utils.criterions import get_criterion
from utils.metrics import get_metrics
from utils import checkpoint


def get_options():
    parser = argparse.ArgumentParser(description="""""")
    parser.add_argument('--run_id', type=str, default=10, help='')
    parser.add_argument("--experiment", type=str, default='test_mpi',
                        help="[default: %(default)s] add experiment.")
    parser.add_argument("--config-file", type=str, help="")
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        arguments = ' '.join(f.readlines()).strip()
        config_list = re.split('\s+', arguments)

    config_list += ['--run_id', str(args.run_id)]

    parser = MainParser()
    options = parser.parse_args(config_list)

    max_len = len(options.__dict__.keys())
    for k, v in options.__dict__.items():
        print("{k:>{x}} {v:}".format(k=k, v=v, x=max_len))

    return options


def main():
    options = get_options()

    options = initialize(options)
    options = create_dataset(options, train=True)
    options = create_dataset(options, train=False)

    model = get_model(options)

    optimizer = get_optimizer(options, model)

    scheduler = get_scheduler(options, optimizer)

    # Criterions are like `torch.nn.CrossEntropyLoss()`
    criterion = get_criterion(options)

    metrics = get_metrics(options)

    if options.use_cuda:
        model.cuda()
        criterion.cuda()

    options = checkpoint.maybe_resume(options, model, optimizer, scheduler)

    controlflow = get_controlflow(options)
    controlflow(model=model, optimizer=optimizer, criterion=criterion,
                metrics=metrics, scheduler=scheduler, options=options)


if __name__ == '__main__':
    main()
