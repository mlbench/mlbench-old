# -*- coding: utf-8 -*-
# import os
import cv2

from parameters import get_args
from mlbench.utils.log import log, log0
from mlbench.utils.set_conf import set_conf
from mlbench.models.create_model import create_model
from mlbench.runs.distributed_running import train_and_validate as train_val_op


def run_mpi(args):
    """distributed training via mpi backend."""
    # Setup config.
    set_conf(args, verbose=True)

    # create model and deploy the model.
    model, criterion, optimizer = create_model(args)

    log0("*" * 80)

    # train amd evaluate model.
    train_val_op(args, model, criterion, optimizer)

    log0("*" * 80)


if __name__ == '__main__':
    args = get_args()
    run_mpi(args)
