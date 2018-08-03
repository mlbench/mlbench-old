# -*- coding: utf-8 -*-

import os
import time
import logging
import logging.config

import torch.distributed as dist

import mlbench.utils.opfiles as opfile


def record(content, path):
    opfile.write_txt(content + "\n", path, type="a")


log_path = None


def configure_log(args=None):
    global log_path

    if args is not None:
        log_path = os.path.join(
            args.checkpoint_dir, 'record' + str(args.graph.rank))
    else:
        log_path = os.path.join(os.getcwd(), "record")


def log(content):
    """print the content while store the information to the path."""
    content = time.strftime("%Y:%m:%d %H:%M:%S") + "\t" + content
    print(content)
    opfile.write_txt(content + "\n", log_path, type="a")


def log0(content):
    if dist.get_rank() == 0:
        log(content)


def setup_logging(log_file='log.txt'):
    """Setup logging configuration."""
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return logging


def logging_computing(args, tracker, loss, prec1, prec5, input):
    # measure accuracy and record loss.
    tracker['losses'].update(loss.item(), input.size(0))
    tracker['top1'].update(prec1[0], input.size(0))
    tracker['top5'].update(prec5[0], input.size(0))

    # measure elapsed time.
    tracker['batch_time'].update(time.time() - tracker['end_data_time'])
    tracker['start_sync_time'] = time.time()


def logging_sync(args, tracker):
    # measure elapsed time.
    tracker['sync_time'].update(time.time() - tracker['start_sync_time'])


def logging_load(args, tracker):
    # measure elapsed time.
    tracker['load_time'].update(time.time() - tracker['start_load_time'])


def logging_display(args, tracker):
    log_info = 'Local index: {local_index}. Load: {load:.3f}s | Data: {data:.3f}s | Batch: {batch:.3f}s | Sync: {sync:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
        local_index=args.local_index,
        load=tracker['load_time'].avg,
        data=tracker['data_time'].avg,
        batch=tracker['batch_time'].avg,
        sync=tracker['sync_time'].avg,
        loss=tracker['losses'].avg,
        top1=tracker['top1'].avg,
        top5=tracker['top5'].avg)
    log('Process {}: '.format(args.graph.rank) + log_info)
    tracker['start_load_time'] = time.time()
