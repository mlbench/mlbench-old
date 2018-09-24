from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch

from optim.lr import SchedulerParser

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
            self.add_argument('--use_cuda', action='store_true', default=False,
                              help="[default: %(default)s] use cuda or not.")

        if communication_backend:
            self.add_argument('--comm_backend', default='mpi', metavar="<CB>",
                              help="[default: %(default)s] backend for distributed pytorch.")

        if run_id:
            self.add_argument('--run_id', type=str, required=True, metavar='<RU>',
                              help="the run_id for the experiment.")

        if resume:
            self.add_argument('--resume', action='store_true', default=False,
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
                 max_train_steps=True, dtype=True, max_batch_per_epoch=True, dont_post_to_dashboard=True):
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
                                   "until the specified number of epochs have run as usual."
                                   "Note that `max_train_steps` should only be used to control the "
                                   "number of epochs to run. Objectis like Learning rate schedule should "
                                   "use `train_epochs`.")

        if dtype:
            self.add_argument("--dtype", type=str, default="fp32", choices=list(DTYPE_MAP.keys()),
                              help="[default: %(default)s] {%(choices)s} The PyTorch datatype "
                              "used for calculations. Variables may be cast to a higher"
                              "precision on a case-by-case basis for numerical stability.",
                              metavar="<DT>")

        if max_batch_per_epoch:
            self.add_argument("--max_batch_per_epoch", type=int, default=None, metavar="<MBPE>",
                              help="[default: %(default)s] maximum number of batchs in one peoch.")

        if dont_post_to_dashboard:
            self.add_argument('--dont_post_to_dashboard', action='store_true', default=False,
                              help='[default: %(default)s] decide whether or not post metrics to dashboard '
                              'if there is one. This can be usefule when lauching the mpi jobs from worker.')


class DatasetParser(argparse.ArgumentParser):
    def __init__(self, add_help=True, batch_size=True, root_data_dir=True, name=True,
                 reshuffle_per_epoch=True, preprocessing_version=True, download_dataset=True,
                 libsvm_dataset=True, sparse_dataset=True, lmdb=True):
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
            self.add_argument("--preprocessing_version", type=str, default="v1", metavar="<PV>",
                              help="[default: %(default)s] versions for preprocessing methods.")

        if reshuffle_per_epoch:
            self.add_argument("--reshuffle_per_epoch", action='store_true', default=True,
                              help="[default: %(default)s] reshuffle the dataset per epoch.")

        if libsvm_dataset:
            self.add_argument("--libsvm_dataset", action='store_true', default=False,
                              help="[default: %(default)s] dataset of LIBSVM format.")

        if sparse_dataset:
            self.add_argument("--sparse_dataset", action='store_true', default=False,
                              help="[default: %(default)s] The dataset contains sparse matrix.")

        if lmdb:
            self.add_argument("--lmdb", action='store_true', default=False,
                              help="[default: %(default)s] The dataset is already in lmdb database. "
                              "root_data_dir is the lmdb database.")


class ModelParser(argparse.ArgumentParser):
    """Arguments related to model, optimizer, lr-scheduler, training, etc."""

    def __init__(self, add_help=True, lr=True, lr_per_sample=True, momentum=True, criterion=True, nesterov=True,
                 weight_decay=True, opt_name=True, model_name=True, model_version=True,
                 lr_scheduler=True, lr_scheduler_level=True, metrics=True):
        super(ModelParser, self).__init__(add_help=add_help)

        lr_group = self.add_mutually_exclusive_group()

        if opt_name:
            self.add_argument("--opt_name", type=str, default='sgd', metavar="<ON>",
                              help="[default: %(default)s] name of the optimizer to use.")

        if model_name:
            self.add_argument("--model_name", type=str, default='resnet18', metavar="<MN>",
                              help="[default: %(default)s] name of the model to use.")

        if model_version:
            self.add_argument("--model_version", type=str, default='default', metavar="<MV>",
                              help="[default: %(default)s] version of model.")

        if lr:
            lr_group.add_argument("--lr", type=float, default=0.1, metavar='<LR>',
                                  help="[default: %(default)s] initial learning rate for the optimizer."
                                  "This learning rate is mutual exclusive with lr_per_sample."
                                  "Note that if warmup is applied, then this lr means the lr after warmup.")

        if lr_per_sample:
            lr_group.add_argument("--lr_per_sample", type=float, default=None, metavar='<LRPS>',
                                  help="[default: %(default)s] initial learning rate per sample for the optimizer."
                                  "This learning rate is mutual exclusive with --lr."
                                  "Note that lr_per_sample is similar to --lr in usage except for the batch size."
                                  "The batch size here refers to --minibatch, not (--minibatch * machines).")

        if lr_scheduler:
            self.add_argument("--lr_scheduler", type=str, default='const', metavar='<LS>',
                              help="[default: %(default)s] name of learning rate scheduler.")

        if lr_scheduler_level:
            self.add_argument("--lr_scheduler_level", type=str, choices=['epoch', 'batch'],
                              default='epoch', metavar='<LSL>',
                              help="[default: %(default)s] scheduling learning rate at epoch/batch level.")

        if momentum:
            self.add_argument("--momentum", type=float, default=0.9, metavar='<M>',
                              help="[default: %(default)s] initial momentum.")

        if criterion:
            self.add_argument("--criterion", type=str, default='CrossEntropyLoss', metavar='<CR>',
                              help="[default: %(default)s] name of training loss function."
                              "Support loss functions from `torch.nn.modules.loss`.")

        if metrics:
            self.add_argument("--metrics", type=str, default='topk', metavar='<ME>',
                              help="[default: %(default)s] The metrics to measure training and evaluations.")

        if nesterov:
            self.add_argument("--nesterov", action='store_true', default=False,
                              help="[default: %(default)s] apply nesterov in the optimizer.")

        if weight_decay:
            self.add_argument("--weight_decay", type=float, default=5e-4, metavar='<WD>',
                              help="[default: %(default)s] weight decay of the optimizer.")


class ControlflowParser(argparse.ArgumentParser):
    def __init__(self, add_help=True, train_epochs=True, epochs_between_evals=True, no_validation=True):
        super(ControlflowParser, self).__init__(add_help=add_help)

        if train_epochs:
            self.add_argument("--train_epochs", type=int, default=1, metavar='<TE>',
                              help="[default: %(default)s] number of epochs to train.")

        if epochs_between_evals:
            self.add_argument("--epochs_between_evals", type=int, default=1, metavar="<EBE>",
                              help="[default: %(default)s] The number of training epochs to run "
                              "between evaluations.")
        if no_validation:
            self.add_argument("--no_validation", action='store_true', default=False,
                              help="[default: %(default)s] Do not perform validation during training.")


class MainParser(argparse.ArgumentParser):
    """An example of main parser."""

    def __init__(self):
        super(MainParser, self).__init__(parents=[
            BaseParser(add_help=False),
            PerformanceParser(add_help=False),
            DatasetParser(add_help=False),
            ModelParser(add_help=False),
            ControlflowParser(add_help=False),
            SchedulerParser(add_help=False)
        ])
