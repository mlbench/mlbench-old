from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


class BaseParser(argparse.ArgumentParser):
    def __init__(self, add_help=True, use_cuda=True, communication_backend=True, run_id=True,
                 resume=True, seed=True, checkpoint=True, cudnn_deterministic=True,
                 checkpoint_root=True, logging_file=True, logging_level=True, debug=True):
        """Arguments decide which parameters are parsed."""
        super(BaseParser, self).__init__(add_help=add_help)

        if use_cuda:
            self.add_argument('--use_cuda', action='store_true', default=True,
                              help="[default: %(default)s] use cuda or not.")

        if communication_backend:
            self.add_argument('--comm_backend', default='mpi', metavar="<CB>",
                              help="[default: %(default)s] backend for distributed pytorch.")

        if run_id:
            self.add_argument('--run_id', type=str, required=True, metavar='<RU>',
                              help="the run_id for the experiment.")

        if resume:
            self.add_argument('--resume', action='store_true', default=True,
                              help="[default: %(default)s] resume experiment specified by run_id.")

        if seed:
            self.add_argument('--seed', type=int, default=42, metavar='<S>',
                              help="[default: %(default)s] set random seed.")

        if checkpoint:
            self.add_argument('--checkpoint', type=str, choices=['best', 'all', 'never'],
                              metavar='<C>', default='never',
                              help="[default: %(default)s] checkpoint model during training.")

        if cudnn_deterministic:
            self.add_argument('--cudnn_deterministic', action='store_true', default=False,
                              help="[default: %(default)s] enable deterministic cudnn training."
                              "WARNING: it may slows down training.")

        if checkpoint_root:
            self.add_argument('--checkpoint_root', type=str, default='/checkpoint', metavar='<CR>',
                              help="[default: %(default)s] root directory of checkpoint path.")

        if logging_file:
            self.add_argument("--logging_file", type=str, default='/mlbench.log', metavar='<LF>',
                              help="[default: %(default)s] save logging message to local file.")

        if logging_level:
            self.add_argument('--logging_level', type=str, default='DEBUG', metavar='<LL>',
                              help="[default: %(default)s] logging level.")

        if debug:
            self.add_argument('--debug', action='store_true', default=False,
                              help="[default: %(default)s] turn on debug mode.")
