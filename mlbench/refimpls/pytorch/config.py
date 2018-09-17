"""Config environments."""
import logging
import os
import random
import torch
import shutil
import socket
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from utils import checkpoint
from utils import log
from utils.topology import FCGraph

logger = logging.getLogger('mlbench')


def config_logging(options):
    """Setup logging modules.

    A stream handler and file handler are added to default logger `mlbench`.
    """

    level = options.logging_level
    logging_file = options.logging_file

    class RankFilter(logging.Filter):
        def filter(self, record):
            record.rank = dist.get_rank()
            return True

    logger = logging.getLogger('mlbench')
    if len(logger.handlers) >= 2:
        return

    logger.setLevel(level)
    logger.addFilter(RankFilter())

    formatter = logging.Formatter(
        '%(asctime)s %(name)s %(rank)s %(levelname)s: %(message)s',
        "%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(logging_file)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def config_pytorch(options):
    """Config pytorch packages.

    Fix random number for packages and initialize distributed environment for pytorch.
    Setup cuda environment for pytorch.

    :param options: A global object containing specified options.
    :type options: argparse.Namespace
    """

    # Setting `cudnn.deterministic = True` will turn on
    # CUDNN deterministic setting which can slow down training considerably.
    # Unexpected behavior may also be observed from checkpoint.
    # See: https: // github.com/pytorch/examples/blob/master/imagenet/main.py
    if options.cudnn_deterministic:
        cudnn.deterministic = True
        log.warning('You have chosen to seed training. '
                    'This will turn on the CUDNN deterministic setting, '
                    'which can slow down your training considerably! '
                    'You may see unexpected behavior when restarting '
                    'from checkpoints.', 0)

    if options.seed is not None:
        random.seed(options.seed)
        torch.manual_seed(options.seed)

    # define the graph for the computation.
    if options.use_cuda:
        assert torch.cuda.is_available()

    options.rank = dist.get_rank()
    options.world_size = dist.get_world_size()
    options.graph = FCGraph(options)

    # enable cudnn accelerator if we are using cuda.
    if options.use_cuda:
        options.graph.assigned_gpu_id()
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        if torch.backends.cudnn.version() is None:
            log.warning("CUDNN not found on device.")

        log.info("World size={}, Rank={}, hostname={}, cuda_available={}, cuda_device={}".format(
            options.world_size, options.rank, socket.gethostname(), torch.cuda.is_available(),
            torch.cuda.current_device()))

        # raise NotImplementedError('Pause')


def config_path(options):
    """Config the path used during the experiments."""

    # Checkpoint for the current run
    options.ckpt_run_dir = checkpoint.get_ckpt_run_dir(
        options.checkpoint_root, options.run_id,
        options.dataset_name, options.model_name, options.opt_name)

    if not options.resume:
        log.info("Remove previous checkpoint directory : {}".format(
            options.ckpt_run_dir))
        shutil.rmtree(options.ckpt_run_dir, ignore_errors=True)
    os.makedirs(options.ckpt_run_dir, exist_ok=True)


def initialize(options):
    """Initialize environment and add additional information."""
    # if not dist._initialized:
    dist.init_process_group(options.comm_backend)

    config_logging(options)

    config_pytorch(options)

    config_path(options)

    return options
