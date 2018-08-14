import copy
from utils import log

from .load_dataset import create_dataset


def display_one_batch(data_loader):
    log.debug("Display one batch of data:")
    for i, (input, target) in enumerate(copy.deepcopy(data_loader)):
        log.debug("\tBatch {}: input.shape={}, target.shape={}".format(
            i, input.shape, target.shape))
        log.debug(input, 0)
        log.debug(target, 0)
        break


def get_dataset(context):
    dataset = context.dataset
    if dataset.train:
        dataset.train_ = create_dataset(
            dataset.name, dataset.root_folder, dataset.batch_size,
            dataset.num_workers, context.meta.rank, context.meta.world_size,
            dataset.reshuffle_per_epoch, context.meta.debug, dataset_type='train')

        # if context.meta.debug:
        #     display_one_batch(dataset.train_.loader)

    if dataset.val:
        dataset.val_ = create_dataset(
            dataset.name, dataset.root_folder, dataset.batch_size,
            dataset.num_workers, context.meta.rank, context.meta.world_size,
            dataset.reshuffle_per_epoch, context.meta.debug, dataset_type='test')
