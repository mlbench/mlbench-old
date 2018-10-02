# -*- coding: utf-8 -*-
"""
This script serialize dataset into LMDB.

For the moment we support LIBSVM format data. One can use flag ``--sparse`` to decide whether or not we convert
features matrix to dense matrix.

For dense matrix:
1. each ``__iter__`` returns a [feature, label] of one sample

For sparse matrix: 
1. each ``__iter__`` returns a [data, row, col, label] of one sample. To recover the dataset, we can use

.. code-block:: python

    from scipy.sparse import coo_matrix
    X = coo_matrix((datas, (rows, cols)), shape=(n_rows, n_cols))

"""
import argparse
import os
import numpy as np

from os.path import join
from sklearn.datasets import load_svmlight_file
from tensorpack.dataflow import dataset, PrefetchDataZMQ, LMDBSerializer

from sklearn.datasets import make_classification


def get_args():
    parser = argparse.ArgumentParser(description='aug data.')

    # define arguments.
    parser.add_argument('--data', default='imagenet', help='a specific dataset name')
    parser.add_argument('--sparse', action='store_true', default=False, help='when the dataset is libsvm format.')
    parser.add_argument('--data_dir', default=None, help='root directory to the dataset')
    parser.add_argument('--data_type', default='train', type=str)

    # parse args.
    args = parser.parse_args()

    # check args.
    assert args.data_dir is not None
    assert args.data_type in ['train', 'val', 'test']
    return args


def build_dirs(path):
    try:
        os.makedirs(path)
    except OSError as e:
        print(e)


"""sequential epsilon or rcv1"""

_DATASET_MAP = {
    'epsilon_train': 'https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2',
    'epsilon_test': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.bz2',
    'rcv1_train': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2',
    'rcv1_test': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2',
    'rcv1_train': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2',
    'url': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/url_combined.bz2',
    'webspam_train':
    'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_trigram.svm.bz2',
    'duke_train': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/duke.bz2',
    'australian_train': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/australian'
}


def maybe_download_and_extract(root, data_url, file_path):
    if not os.path.exists(root):
        os.makedirs(root)

    file_name = data_url.split('/')[-1]

    if len([x for x in os.listdir(root) if x == file_name]) == 0:
        os.system('wget -t inf {} -O {}'.format(data_url, file_path))


def _get_dense_tensor(tensor):
    if 'sparse' in str(type(tensor)):
        return tensor.toarray()
    elif 'numpy' in str(type(tensor)):
        return tensor


def _correct_binary_labels(labels, is_01_classes=True):
    classes = set(labels)

    if -1 in classes and is_01_classes:
        labels[labels == -1] = 0
    return labels


class LIBSVMDataset(object):
    def __init__(self, root, name, split, is_sparse):
        self.is_sparse = is_sparse

        # get file url and file path.
        data_url = _DATASET_MAP['{}_{}'.format(name, split)]
        file_path = os.path.join(root, data_url.split('/')[-1])

        # download dataset or not.
        maybe_download_and_extract(root, data_url, file_path)

        # load dataset.
        dataset = load_svmlight_file(file_path)
        self.features, self.labels = self._get_features_and_labels(dataset)

    def _get_features_and_labels(self, data):
        features, labels = data

        features = _get_dense_tensor(features) if not self.is_sparse else features
        labels = _get_dense_tensor(labels)
        labels = _correct_binary_labels(labels)
        return features, labels

    def __len__(self):
        return self.features.shape[0]

    def __iter__(self):
        idxs = list(range(self.__len__()))
        for k in idxs:
            features = self.features[k]
            label = self.labels[k]
            if self.is_sparse:
                features = features.tocoo()
                yield [(features.data, features.row, features.col), label]
            else:
                yield [features, label]

    def get_data(self):
        return self.__iter__()

    def size(self):
        return self.__len__()

    def reset_state(self):
        """
        Reset state of the dataflow.
        It **has to** be called once and only once before producing datapoints.
        Note:
            1. If the dataflow is forked, each process will call this method
               before producing datapoints.
            2. The caller thread of this method must remain alive to keep this dataflow alive.
        For example, RNG **has to** be reset if used in the DataFlow,
        otherwise it won't work well with prefetching, because different
        processes will have the same RNG state.
        """
        pass


class SyntheticLIBSVMDataset(object):
    def __init__(self, features, labels):
        self.features, self.labels = features, labels

    def __len__(self):
        return self.features.shape[0]

    def __iter__(self):
        idxs = list(range(self.__len__()))
        for k in idxs:
            features = self.features[k]
            label = [self.labels[k]]
            yield [features, label]

    def get_data(self):
        return self.__iter__()

    def size(self):
        return self.__len__()

    def reset_state(self):
        """
        Reset state of the dataflow.
        It **has to** be called once and only once before producing datapoints.
        Note:
            1. If the dataflow is forked, each process will call this method
               before producing datapoints.
            2. The caller thread of this method must remain alive to keep this dataflow alive.
        For example, RNG **has to** be reset if used in the DataFlow,
        otherwise it won't work well with prefetching, because different
        processes will have the same RNG state.
        """
        pass


def sequential_epsilon_or_rcv1(root_path, name, data_type, is_sparse):
    data = LIBSVMDataset(root_path, name, data_type, is_sparse)
    lmdb_file_path = join(root_path, '{}_{}.lmdb'.format(name, data_type))

    print('dump_dataflow_to_lmdb for {}'.format(lmdb_file_path))
    ds1 = PrefetchDataZMQ(data, nr_proc=1)
    LMDBSerializer.save(ds1, lmdb_file_path)


def sequential_synthetic_dataset(root_path, dataset_name):
    """Generate a synthetic dataset for regression."""
    if 'dense' in dataset_name:
        X, y = make_classification(n_samples=10000, n_features=100, n_informative=90, n_classes=2, random_state=42)
    else:
        raise NotImplementedError("{} synthetic dataset is not supported.".format(dataset_name))

    data = SyntheticLIBSVMDataset(X, y)
    lmdb_file_path = join(root_path, '{}.lmdb'.format(dataset_name))

    print('dump_dataflow_to_lmdb for {}'.format(lmdb_file_path))
    ds1 = PrefetchDataZMQ(data, nr_proc=1)
    LMDBSerializer.save(ds1, lmdb_file_path)


def main(args):
    if 'epsilon' in args.data or 'rcv1' in args.data or 'webspam' in args.data:
        sequential_epsilon_or_rcv1(args.data_dir, args.data, args.data_type, args.sparse)
    elif 'australian' in args.data or 'duke' in args.data:
        # These two are small datasets for testing purpose.
        sequential_epsilon_or_rcv1(args.data_dir, args.data, args.data_type, args.sparse)
    elif 'synthetic' in args.data:
        sequential_synthetic_dataset(args.data_dir, args.data)
    else:
        raise NotImplementedError("Dataset {} not supported.".format(args.data))


if __name__ == '__main__':
    args = get_args()
    main(args)

# python code/dataset/sequential_data.py --data_dir /mlodata1/dl-dataset/data/imagenet/ --data_type train
# python code/dataset/sequential_data.py --data_dir /mlodata1/dl-dataset/data/imagenet/ --data_type val
