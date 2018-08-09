"""
colors: https://stackoverflow.com/questions/5947742/how-to-change-the-output-color-of-echo-in-linux
"""
import logging
import torch.distributed as dist

logger = logging.getLogger('mlbench')


def info(content):
    logger.info(content)


def debug(content):
    logger.debug(content)


def warning(content):
    logger.warning("\033[0;31m{}\033[0m".format(content))


def critical(content):
    logger.critical("\033[0;104m{}\033[0m".format(content))


def todo(content):
    logger.warning("\033[0;33m{}\033[0m".format(content))
