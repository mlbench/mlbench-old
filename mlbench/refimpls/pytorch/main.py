from config import initialize

from utils.parser import MainParser
from datasets.load_dataset import create_dataset
from models import get_model
from optim.optimizer import get_optimizer
from optim.lr import get_scheduler
from utils.metrics import get_metrics
from utils.criterions import get_criterion
from controlflow.controlflow import get_controlflow
from utils import checkpoint


def main():
    parser = MainParser()
    options = parser.parse_args()
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
        # TODO: add accuracy metrics to cuda?

    options = checkpoint.maybe_resume(options, model, optimizer, scheduler)

    controlflow = get_controlflow(options)
    controlflow(model=model, optimizer=optimizer, criterion=criterion,
                metrics=metrics, scheduler=scheduler, options=options)


if __name__ == '__main__':
    main()
