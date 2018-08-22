from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch

DTYPE_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}


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


class PerformanceParser(argparse.ArgumentParser):
    def __init__(self, add_help=True, num_parallel_workers=True, use_synthetic_data=True,
                 max_train_steps=True, dtype=True):
        super(PerformanceParser, self).__init__(add_help=add_help)

        if num_parallel_workers:
            self.add_argument("--num_parallel_workers", "-npw", type=int, default=4,
                              help="[default: %(default)s] The number of records that are "
                              "processed in parallel  during input processing. This can be "
                              "optimized per data set but for generally homogeneous data "
                              "sets, should be approximately the number of available CPU "
                              "cores.",
                              metavar="<NPW>")

        if use_synthetic_data:
            self.add_argument("--use_synthetic_data", action="store_true", default=False,
                              help="[default: %(default)s] If set, use fake data (zeroes) instead of a real dataset. "
                              "This mode is useful for performance debugging, as it removes "
                              "input processing steps, but will not learn anything.")

        if max_train_steps:
            self.add_argument("--max_train_steps", type=int, default=None, metavar="<MTS>",
                              help="[default: %(default)s] The model will stop training if the "
                                   "global_step reaches this value. If not set, training will run"
                                   "until the specified number of epochs have run as usual. It is"
                                   "generally recommended to set --train_epochs=1 when using this"
                                   "flag.")

        if dtype:
            self.add_argument("--dtype", type=str, default="fp32", choices=list(DTYPE_MAP.keys()),
                              help="[default: %(default)s] {%(choices)s} The PyTorch datatype "
                              "used for calculations. Variables may be cast to a higher"
                              "precision on a case-by-case basis for numerical stability.",
                              metavar="<DT>")


class DatasetParser(argparse.ArgumentParser):
    def __init__(self, add_help=True, batch_size=True, root_data_dir=True, name=True,
                 reshuffle_per_epoch=True, preprocessing_version=True, download_dataset=True):
        super(DatasetParser, self).__init__(add_help=add_help)

        if batch_size:
            self.add_argument("--batch_size", type=int, default=32, metavar="<BS>",
                              help="[default: %(default)s] default batch size for training and evaluation.")

        if root_data_dir:
            self.add_argument("--root_data_dir", type=str, default="/datasets/torch", metavar="<DD>",
                              help="[default: %(default)s] root directory to all datasets."
                              "If the given dataset name, its directory is root_data_dir/dataset_name.")

        if name:
            self.add_argument("--dataset_name", type=str, default='mnist', metavar="<DN>",
                              help="[default: %(default)s] the dataset name for training.")

        if download_dataset:
            self.add_argument("--download_dataset", action="store_true", default=True,
                              help="[default: %(default)s] allow download dataset.")

        if preprocessing_version:
            self.add_argument("--preprocessing_version", type=str, default="default", metavar="<PV>",
                              help="[default: %(default)s] versions for preprocessing methods.")

        if reshuffle_per_epoch:
            self.add_argument("--reshuffle_per_epoch", action='store_true', default=True,
                              help="[default: %(default)s] reshuffle the dataset per epoch.")


class ModelParser(argparse.ArgumentParser):
    """Arguments related to model, optimizer, lr-scheduler, training, etc."""

    def __init__(self, add_help=True, lr=True, momentum=True, criterion=True, nesterov=True,
                 weight_decay=True, opt_name=True, model_name=True, model_version=True):
        super(ModelParser, self).__init__(add_help=add_help)

        if opt_name:
            self.add_argument("--opt_name", type=str, default='sgd', metavar="<ON>",
                              help="[default: %(default)s] name of the optimizer to use.")

        if model_name:
            self.add_argument("--model_name", type=str, default='resnet18', metavar="<MN>",
                              help="[default: %(default)s] name of the model to use.")

        if model_version:
            self.add_argument("--model_version", type=str, default='v1', metavar="<MV>",
                              help="[default: %(default)s] version of model.")

        if lr:
            self.add_argument("--lr", type=float, default=0.1, metavar='<LR>',
                              help="[default: %(default)s] initial learning rate.")

        if momentum:
            self.add_argument("--momentum", type=float, default=0.9, metavar='<M>',
                              help="[default: %(default)s] initial momentum.")

        if criterion:
            self.add_argument("--criterion", type=str, default='CrossEntropyLoss', metavar='<CR>',
                              help="[default: %(default)s] name of training loss function."
                              "Support loss functions from `torch.nn.modules.loss`.")

        if nesterov:
            self.add_argument("--nesterov", action='store_true', default=True,
                              help="[default: %(default)s] apply nesterov in the optimizer.")

        if weight_decay:
            self.add_argument("--weight_decay", type=float, default=5e-4, metavar='<WD>',
                              help="[default: %(default)s] weight decay of the optimizer.")


class ControlflowParser(argparse.ArgumentParser):
    def __init__(self, add_help=True, train_epochs=True, epochs_between_evals=True):
        super(ControlflowParser, self).__init__(add_help=add_help)

        if train_epochs:
            self.add_argument("--train_epochs", type=int, default=1, metavar='<TE>',
                              help="[default: %(default)s] number of epochs to train.")

        if epochs_between_evals:
            self.add_argument("--epochs_between_evals", type=int, default=1, metavar="<EBE>",
                              help="[default: %(default)s] The number of training epochs to run "
                              "between evaluations.")


class MainParser(argparse.ArgumentParser):
    """An example of main parser."""

    def __init__(self):
        super(MainParser, self).__init__(parents=[
            BaseParser(add_help=False),
            PerformanceParser(add_help=False),
            DatasetParser(add_help=False),
            ModelParser(add_help=False),
            ControlflowParser(add_help=False)
        ])
