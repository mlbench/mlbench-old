import os
import shutil
import torch

from utils import communication as comm
from utils import log


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
    """Determine the checkpoint path to restore."""
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
        raise FileNotFoundError("Found {} ; Expect {}".format(found_ckpts, ))


def save(options, model, optimizer, scheduler, is_best):
    if options.checkpoint == 'never':
        return

    state = {
        'options_runtime': options.runtime,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }

    dirname = options.ckpt_run_dir
    filename = get_ckpt_id(options.runtime['current_epoch'], options.rank)
    checkpoint_path = os.path.join(dirname, filename)
    best_model_path = os.path.join(dirname, 'model_best.pth.tar')

    if options.checkpoint == 'all':
        torch.save(state, checkpoint_path)
        if is_best:
            shutil.copyfile(checkpoint_path, best_model_path)
    elif options.checkpoint == 'best':
        torch.save(state, best_model_path)
    else:
        raise NotImplementedError


def resume(options, model, optimizer, scheduler):
    checkpoint_path = determine_restore_ckpt_path(
        options.rank, options.checkpoint_root, options.run_id)

    log.info('Try to load previous model from the path:{}'.format(checkpoint_path))
    if os.path.isfile(checkpoint_path):
        # get checkpoint.
        checkpoint = torch.load(checkpoint_path)
        # restore some run-time info.

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        # logging.
        log.info("Loaded checkpoint '{}' (epoch {})".format(
            checkpoint_path, checkpoint['options_runtime']['current_epoch']))
        checkpoint['options_runtime']['current_epoch'] = checkpoint['options_runtime']['current_epoch'] + 1
    else:
        raise FileNotFoundError("No checkpoint found at '{}'".format(options.resume))
    return checkpoint['options_runtime']


def maybe_resume(options, model, optimizer, scheduler):
    """Recover the state of options, model, optimizer and scheduler."""
    if options.resume:
        # reload model from the latest checkpoint.
        options.runtime = resume(options, model, optimizer, scheduler)
    else:
        options.runtime = {}
    return options
