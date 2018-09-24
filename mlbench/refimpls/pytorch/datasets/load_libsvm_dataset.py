import lmdb
import os
import torch.utils.data
import pickle
import numpy as np
import math

from tensorpack.utils.compatible_serialize import loads
from tensorpack.dataflow.serialize import LMDBSerializer

from .libsvm_dataloader import numpy_sparse_collate
from .partition_data import DataPartitioner

__all__ = ['load_libsvm_lmdb', 'IMDBPT']

_LIBSVM_DATASETS = [
    {'name': 'webspam', 'n_samples': 350000, 'n_features': 16609143, 'sparse': True},
    {'name': 'epsilon-train', 'n_samples': 400000, 'n_features': 2000, 'sparse': False},
    {'name': 'duke-train', 'n_samples': 44, 'n_features': 7129, 'sparse': True},
    {'name': 'australian-train', 'n_samples': 690, 'n_features': 14, 'sparse': False},
    {'name': 'rcv1-train', 'n_samples': 677399, 'n_features': 47236, 'sparse': True}
]


class IMDBPT(torch.utils.data.Dataset):
    """
    Args:
        root (string): Either root directory for the database files,
            or a absolute path pointing to the file.
        classes (string or list): One of {'train', 'val', 'test'} or a list of
            categories to load. e,g. ['bedroom_train', 'church_train'].
        transform (callable, optional): A function/transform that
            takes in an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
        target_transform (callable, optional):
            A function/transform that takes in the target and transforms it.
    """

    def __init__(self, root, transform=None, target_transform=None, is_image=True, n_features=None):
        self.n_features = n_features
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.lmdb_files = self._get_valid_lmdb_files()

        # for each class, create an LSUNClassDataset
        self.dbs = []
        for lmdb_file in self.lmdb_files:
            self.dbs.append(LMDBPTClass(
                root=lmdb_file, transform=transform,
                target_transform=target_transform, is_image=is_image))

        # build up indices.
        self.indices = np.cumsum([len(db) for db in self.dbs])
        self.length = self.indices[-1]
        self._get_index_zones = self._build_indices()

    def _get_valid_lmdb_files(self):
        """get valid lmdb based on given root."""
        if not self.root.endswith('.lmdb'):
            for l in os.listdir(self.root):
                if '_' in l and '-lock' not in l:
                    yield os.path.join(self.root, l)
        else:
            yield self.root

    def _build_indices(self):
        indices = self.indices
        from_to_indices = enumerate(zip(indices[: -1], indices[1:]))

        def f(x):
            if len(list(from_to_indices)) == 0:
                return 0, x

            for ind, (from_index, to_index) in from_to_indices:
                if from_index <= x and x < to_index:
                    return ind, x - from_index
        return f

    def _get_matched_index(self, index):
        return self._get_index_zones(index)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target)
        """
        block_index, item_index = self._get_matched_index(index)
        image, target = self.dbs[block_index][item_index]
        return image, target

    def __len__(self):
        return self.length

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp,
            self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp,
            self.target_transform.__repr__().replace(
                '\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class LMDBPTClass(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, is_image=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.is_image = is_image

        # open lmdb env.
        self.env = self._open_lmdb()

        # get file stats.
        self._get_length()

        # prepare cache_file
        self._prepare_cache()

    def _open_lmdb(self):
        return lmdb.open(
            self.root,
            subdir=os.path.isdir(self.root),
            readonly=True, lock=False, readahead=False,
            map_size=1099511627776 * 2,
            max_readers=1, meminit=False)

    def _get_length(self):
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

            if txn.get(b'__keys__') is not None:
                self.length -= 1

    def _prepare_cache(self):
        cache_file = self.root + '_cache_'
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key
                             for key, _ in txn.cursor() if key != b'__keys__']
            pickle.dump(self.keys, open(cache_file, "wb"))

    def _image_decode(self, x):
        image = cv2.imdecode(x, cv2.IMREAD_COLOR).astype('uint8')
        return Image.fromarray(image, 'RGB')

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            bin_file = txn.get(self.keys[index])

        image, target = loads(bin_file)
        if self.is_image:
            image = self._image_decode(image)

        # print('image', image)
        if self.transform is not None:
            image = self.transform(image)

        # print('self.transform', self.transform)
        # print('image', image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.root + ')'


def construct_sparse_matrix(triplet, n_features):
    from scipy.sparse import coo_matrix, csr_matrix
    data, row, col = triplet
    mat = coo_matrix((data, (row, col)), shape=(len(set(row)), n_features))
    return csr_matrix(mat)[list(set(row))]


def maybe_transform_sparse(stats):
    return (lambda x: construct_sparse_matrix(x, stats['n_features'])) \
        if stats['sparse'] else None


def get_dataset_info(name):
    stats = list(filter(lambda x: x['name'] == name, _LIBSVM_DATASETS))
    assert len(stats) == 1, '{} not found.'.format(name)
    return stats[0]


def load_libsvm_lmdb(name, lmdb_path):
    stats = get_dataset_info(name)
    # print('maybe_transform_sparse(stats)', maybe_transform_sparse(stats))

    dataset = IMDBPT(lmdb_path, transform=maybe_transform_sparse(stats), is_image=False)
    return dataset


def partition_dataset(name, root_folder, batch_size, num_workers, rank, world_size,
                      reshuffle_per_epoch, preprocessing_version, train=True, download=True, pin_memory=True):
    """ Load a partition of dataset from by the rank. """
    dataset = load_libsvm_lmdb(name, root_folder)

    # Partition dataset and use the one corresponding to `rank`.
    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    partition = DataPartitioner(dataset, rank, reshuffle_per_epoch, partition_sizes)
    data_to_load = partition.use(rank)
    num_samples_per_device = len(data_to_load)

    # create a data loader.
    data_loader = torch.utils.data.DataLoader(
        data_to_load, batch_size=batch_size, shuffle=train, collate_fn=numpy_sparse_collate,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

    info = {'loader': data_loader,
            'num_samples_per_device': num_samples_per_device,
            'num_batches': math.ceil(1.0 * num_samples_per_device / batch_size)}

    if hasattr(dataset, 'classes'):
        info['num_classes'] = len(dataset.classes)
    return info


def create_dataset(options, train=True):
    """Create a dataset and add information to options."""
    dataset = partition_dataset(name=options.dataset_name,
                                root_folder=options.root_data_dir,
                                batch_size=options.batch_size,
                                num_workers=options.num_parallel_workers,
                                rank=options.rank,
                                world_size=options.world_size,
                                reshuffle_per_epoch=options.reshuffle_per_epoch,
                                train=train,
                                preprocessing_version=options.preprocessing_version,
                                pin_memory=options.use_cuda)

    if 'num_classes' in dataset:
        options.num_classes = dataset['num_classes']

    if train:
        options.train_loader = dataset['loader']
        options.train_num_samples_per_device = dataset['num_samples_per_device']
        options.train_num_batches = dataset['num_batches']
    else:
        options.val_loader = dataset['loader']
        options.val_num_samples_per_device = dataset['num_samples_per_device']
        options.val_num_batches = dataset['num_batches']
    return options
