import os
import logging
import torch
import torch.distributed as dist

from mlbench.utils.topology import FCGraph
logger = logging.getLogger('mlbench')


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Context(object):
    def __init__(self, optimizer, dataset, model, controlflow, meta):
        self.optimizer = optimizer
        self.dataset = dataset
        self.model = model
        self.controlflow = controlflow
        self.meta = meta


def _init_context(args):
    config_file = args.config_file

    meta = {
        'logging_level': eval('logging.' + args.logging_level.upper()),
        'logging_file': 'mlbench.log',
        'checkpoint_root': 'checkpoint',
        'checkpoint_overwrite': True,
        'config_id': 'default',
        'use_cuda': False,
        'backend': 'mpi',
        'manual_seed': 42,
    }

    default_optimizer = {
        "name": "sgd",
        "lr": 0.01
    }

    default_controlflow = {
        'name': 'train'
    }

    default_dataset = {
        'name': 'mnist'
    }

    default_model = {
        'name': 'twolayer'
    }

    if config_file is not None:
        with open(config_file, 'r') as f:
            config = json.load(f)

        default_optimizer.update(config.get('optimizer', {}))
        default_controlflow.update(config.get('workflow', {}))
        default_dataset.update(config.get('dataset', {}))
        default_model.update(config.get('model', {}))

    return Context(AttrDict(default_optimizer), AttrDict(default_dataset), AttrDict(default_model),
                   AttrDict(default_controlflow), AttrDict(meta))


def config_logging(context):
    """Setup logging modules."""
    level = context.meta.logging_level
    logging_file = context.meta.logging_file

    class RankFilter(logging.Filter):
        def filter(self, record):
            record.rank = dist.get_rank()
            return True

    logger = logging.getLogger('mlbench')
    logger.setLevel(level)
    logger.addFilter(RankFilter())

    formatter = logging.Formatter('%(asctime)s %(name)s %(rank)s %(levelname)s: %(message)s',
                                  "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(logging_file)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def config_pytorch(meta):
    dist.init_process_group(meta.backend)

    # Set manual seed for both cpu and cuda
    torch.manual_seed(meta.manual_seed)

    # define the graph for the computation.
    if meta.use_cuda:
        assert torch.cuda.is_available()

    meta.rank = dist.get_rank()
    meta.world_size = dist.get_world_size()
    meta.graph = FCGraph(meta)

    if meta.use_cuda:
        torch.cuda.set_device(meta.graph.assigned_gpu_id)

    # enable cudnn accelerator if we are using cuda.
    if meta.use_cuda:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    logger.warning("Add graph into meta and determine device.")

    # local conf.
    # set_local_stat(args)
    logger.warning("Add set_local_stat.")


def config_path(context):
    """After initialization, update some information."""

    # Checkpoint for the current run
    context.meta.checkpoint_run_root = os.path.join(
        context.meta.checkpoint_root, context.dataset.name,
        context.model.name, context.meta.config_id)

    context.meta.checkpoint_run_rank = os.path.join(
        context.meta.checkpoint_run_root, str(context.meta.graph.rank))

    os.makedirs(context.meta.checkpoint_run_rank, exist_ok=context.meta.checkpoint_overwrite)

    # maybe create directories for logging file in the future.


def init_context(args):
    # Context build from args, file and defaults
    context = _init_context(args)

    config_logging(context)

    config_pytorch(context.meta)

    # Customize configuration based on meta information.
    config_path(context)

    return context
