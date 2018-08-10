from mlbench.utils import log
import torch.nn as nn


def distributed_running(controlflow, model, optimizer, criterion, metrics, context):
    log.centering("LAUNCH distributed_running")

    controlflow(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        metrics=metrics,
        data_loader=context.dataset.train_.loader,
        num_epochs=context.controlflow.num_epochs,
        num_batches=context.dataset.train_.num_batches,
        batch_size=context.dataset.batch_size,
        start_epoch=context.controlflow.start_epoch,
        use_cuda=context.meta.use_cuda,
        avg_model=context.controlflow.avg_model,
        world_size=context.meta.world_size,
        debug=context.meta.debug)
