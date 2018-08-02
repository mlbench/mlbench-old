# -*- coding: utf-8 -*-
import argparse
import os
from os.path import join


def get_args():
    parser = argparse.ArgumentParser(description='aug data.')

    # define arguments.
    parser.add_argument('--data', default='imagenet',
                        help='a specific dataset name')
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--data_type', default='train', type=str)

    # parse args.
    args = parser.parse_args()

    # check args.
    assert args.data_dir is not None
    assert args.data_type == 'train' or args.data_type == 'val'
    return args


def build_dirs(path):
    try:
        os.makedirs(path)
    except OSError as e:
        print(e)


def sequential_data(root_path):
    # define path.
    lmdb_path = join(root_path, 'lmdb')
    build_dirs(lmdb_path)
    lmdb_file_path = join(lmdb_path, args.data_type + '.lmdb')

    # build lmdb data.
    from tensorpack.dataflow import dataset, PrefetchDataZMQ, dftools
    import numpy as np

    class BinaryILSVRC12(dataset.ILSVRC12Files):
        def get_data(self):
            for fname, label in super(BinaryILSVRC12, self).get_data():
                with open(fname, 'rb') as f:
                    jpeg = f.read()
                jpeg = np.asarray(bytearray(jpeg), dtype='uint8')
                yield [jpeg, label]

    ds0 = BinaryILSVRC12(root_path, args.data_type, shuffle=False)
    ds1 = PrefetchDataZMQ(ds0, nr_proc=1)
    dftools.dump_dataflow_to_lmdb(ds1, lmdb_file_path)


def main(args):
    if args.data == 'imagenet':
        root_path = args.data_dir

    sequential_data(root_path)


if __name__ == '__main__':
    args = get_args()
    main(args)

# python code/dataset/sequential_data.py --data_dir /mlodata1/dl-dataset/data/imagenet/ --data_type train
# python code/dataset/sequential_data.py --data_dir /mlodata1/dl-dataset/data/imagenet/ --data_type val
