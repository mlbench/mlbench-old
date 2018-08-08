# -*- coding: utf-8 -*-
import logging
import torch.distributed as dist

logger = logging.getLogger('mlbench')


def log(content):
    """print the content while store the information to the path."""
    logger.info(content)


def log0(content):
    """Log content at rank."""
    if dist.get_rank() == 0:
        log(content)


class RankFilter(logging.Filter):
    def filter(self, record):
        record.rank = dist.get_rank()
        return True


def config_logging(args):
    """Setup logging modules."""
    level = logging.DEBUG

    logger = logging.getLogger('mlbench')
    logger.setLevel(level)
    logger.addFilter(RankFilter())

    formatter = logging.Formatter('%(asctime)s %(name)s %(rank)s %(levelname)s: %(message)s', "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(args.log_dir)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
