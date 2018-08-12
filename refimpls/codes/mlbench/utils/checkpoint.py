import os
import torch
import shutil
import torch.distributed as dist
from mlbench.utils import log
from mlbench.utils import communication as comm


def get_ckpt_run_dir(checkpoint_root, run_id, dataset_name, model_name, optimizer_name):
    if isinstance(run_id, str):
        assert '_' not in run_id
    run_dir = "{run_id}_{dataset}_{model}_{optimizer}".format(
        run_id=run_id, dataset=dataset_name, model=model_name, optimizer=optimizer_name)
    return os.path.join(checkpoint_root, run_dir)


def get_ckpt_id(epoch, rank):
    # {epoch}_{batch} can be sorted
    return "{epoch}_{rank}.pth.tar".format(epoch=epoch, rank=rank)


def determine_restore_ckpt_path(rank, checkpoint_root, run_id):
    """Determine the checkpoint path to restore.
    """
    ckpt_run_dirs = os.listdir(checkpoint_root)

    # parse run_ids
    found_ckpts = list(filter(lambda x: x.split("_", 1)[0] == str(run_id), ckpt_run_dirs))

    if len(found_ckpts) == 1:
        found_ckpts = found_ckpts[0]

        ckpt_ids = os.listdir(os.path.join(checkpoint_root, found_ckpts))
        ckpt_ids = list(set(ckpt_ids) - set(['model_best.pth.tar']))

        latest = sorted(map(lambda x: x.split("_")[:2], ckpt_ids))[-1]
        latest = comm.elementwise_min(torch.tensor([int(latest[0])]))
        epoch = latest[0]

        path = os.path.join(checkpoint_root, found_ckpts, get_ckpt_id(epoch, rank))
        return path
    else:
        raise FileNotFoundError(found_ckpts)


def save(state, is_best, context):
    dirname = context.meta.ckpt_run_dir
    filename = get_ckpt_id(context.runtime.current_epoch, context.meta.rank)

    checkpoint_path = os.path.join(dirname, filename)
    best_model_path = os.path.join(dirname, 'model_best.pth.tar')
    torch.save(state, checkpoint_path)

    if is_best:
        shutil.copyfile(checkpoint_path, best_model_path)


def resume(context, model, optimizer):
    if context.meta.resume:
        # reload model from the latest checkpoint.
        checkpoint_index = ''
        checkpoint_path = determine_restore_ckpt_path(
            context.meta.rank, context.meta.checkpoint_root, context.meta.run_id)

        log.info('Try to load previous model from the path:{}'.format(checkpoint_path))
        if os.path.isfile(checkpoint_path):
            # get checkpoint.
            checkpoint = torch.load(checkpoint_path)

            context = checkpoint['context']

            # restore some run-time info.
            context.controlflow.start_epoch = checkpoint['current_epoch'] + 1

            # restore model.
            model.load_state_dict(checkpoint['state_dict'])
            # restore optimizer.
            optimizer.load_state_dict(checkpoint['optimizer'])
            # logging.
            log.info("Loaded checkpoint '{}' (epoch {})".format(
                context.meta.resume, checkpoint['current_epoch']))
        else:
            raise FileNotFoundError("No checkpoint found at '{}'".format(context.meta.resume))

    return context
