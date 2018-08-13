"""
colors: https://stackoverflow.com/questions/5947742/how-to-change-the-output-color-of-echo-in-linux
"""
import logging
import torch.distributed as dist

logger = logging.getLogger('mlbench')


def _warp(string, symbol='*', length=80):
    one_side_length = (length - len(string) - 2) // 2
    if one_side_length > 0:
        return symbol * one_side_length + ' ' + string + ' ' + symbol * one_side_length
    else:
        return string


def centering(content, who='all', symbol='*', length=80):
    info(_warp(content, symbol, length), who=who)


def info(content, who='all'):
    if who == 'all' or who == dist.get_rank():
        logger.info(content)


def debug(content, who='all'):
    if who == 'all' or who == dist.get_rank():
        logger.debug(content)


def warning(content, who='all'):
    if who == 'all' or who == dist.get_rank():
        logger.warning("\033[0;31m{}\033[0m".format(content))


def critical(content, who='all'):
    if who == 'all' or who == dist.get_rank():
        logger.critical("\033[0;104m{}\033[0m".format(content))


def todo(content, who='all'):
    if who == 'all' or who == dist.get_rank():
        logger.warning("\033[0;33m{}\033[0m".format(content))


def configuration_information(context):
    centering("Configuration Information", 0)

    centering('Opimizer', 0, symbol='=', length=40)
    context.log('optimizer')

    centering('Model', 0, symbol='=', length=40)
    context.log('model')

    centering('Dataset', 0, symbol='=', length=40)
    context.log('dataset')

    centering('Controlflow', 0, symbol='=', length=40)
    context.log('controlflow')

    centering("START TRAINING", 0)
