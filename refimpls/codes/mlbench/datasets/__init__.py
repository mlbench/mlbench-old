from mlbench.utils import log

from .load_dataset import create_dataset


def get_dataset(context):
    log.debug("Loading dataset...")
    train_loader, val_loader = create_dataset(
        context.dataset.name, context.dataset.root_folder,
        context.dataset.batch_size, context.dataset.num_workers,
        context.meta.rank)

    log.todo('TODO: Avoid using train_loader/val_loader directly.')
    context.dataset.train_loader = train_loader
    context.dataset.val_loader = val_loader
    log.debug("Dataset loaded...")
