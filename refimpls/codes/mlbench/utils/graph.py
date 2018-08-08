# -*- coding: utf-8 -*-
from abc import abstractmethod, ABCMeta
import numpy as np
import scipy as sp
import random
from scipy.sparse.linalg import eigs
from scipy.sparse import coo_matrix


class UndirectedGraph(metaclass=ABCMeta):

    @property
    @abstractmethod
    def n_edges(self):
        pass

    @property
    @abstractmethod
    def n_nodes(self):
        pass

    @property
    @abstractmethod
    def beta(self):
        pass

    @property
    @abstractmethod
    def matrix(self):
        pass

    @abstractmethod
    def get_neighborhood(self, node_id):
        pass


class CompleteGraph(UndirectedGraph):
    def __init__(self, n):
        self.n = n
        self._data = np.ones((self.n, self.n)) / self.n

    @property
    def n_edges(self):
        return self.n * (self.n - 1) / 2

    @property
    def n_nodes(self):
        return self.n

    @property
    def beta(self):
        return 0

    @property
    def matrix(self):
        return self._data

    def get_neighborhood(self, node_id):
        row = self._data[node_id]
        return {c: v for c, v in zip(range(len(row)), row)}


class RingGraph(UndirectedGraph):
    def __init__(self, n):
        self.n = n
        self._data, self._beta = self._compute_beta(n)

    def _compute_beta(self, n):
        assert n > 2

        # create ring matrix
        diag_rows = np.array(
            [[1/3 for _ in range(n)],
             [1/3 for _ in range(n)],
             [1/3 for _ in range(n)]])
        positions = [-1, 0, 1]
        data = sp.sparse.spdiags(diag_rows, positions, n, n).tolil()
        data[0, n-1] = 1/3
        data[n-1, 0] = 1/3
        data = data.tocsr()

        if n > 3:
            # Find largest real part
            eigenvalues, _ = eigs(data, k=2, which='LR')
            lambda2 = min(abs(i.real) for i in eigenvalues)

            # Find smallest real part
            eigenvalues, _ = eigs(data, k=1, which='SR')
            lambdan = eigenvalues[0].real
        else:
            eigenvals = sorted(np.linalg.eigvals(data.toarray()), reverse=True)
            lambda2 = eigenvals[1]
            lambdan = eigenvals[-1]

        beta = max(abs(lambda2), abs(lambdan))
        return data, beta

    @property
    def n_edges(self):
        return self.n

    @property
    def n_nodes(self):
        return self.n

    @property
    def beta(self):
        return self._beta

    @property
    def matrix(self):
        return self._data

    def get_neighborhood(self, node_id):
        row = self._data.getrow(node_id)
        _, cols = row.nonzero()
        vals = row.data
        return {int(c): v for c, v in zip(cols, vals)}


class NonUniformWeightRingGraph(RingGraph):
    """
    For node i which connects to node_{i-1}, node_{i}, node_{i+1},
    the weight is given by
        node_{i} : x
        node_{i-1} : (1 - x) / 2
        node_{i-1} : (1 + x) / 2
    """

    def __init__(self, n, local_percent):
        self.w_local = local_percent
        self.w_neigh = (1 - local_percent) / 2
        super(NonUniformWeightRingGraph, self).__init__(n)

    def _compute_beta(self, n):
        assert n > 2

        # create ring matrix
        diag_rows = np.array(
            [[self.w_neigh for _ in range(n)],
             [self.w_local for _ in range(n)],
             [self.w_neigh for _ in range(n)]])
        positions = [-1, 0, 1]
        data = sp.sparse.spdiags(diag_rows, positions, n, n).tolil()
        data[0, n-1] = self.w_neigh
        data[n-1, 0] = self.w_neigh
        data = data.tocsr()

        if n > 3:
            # Find largest real part
            eigenvalues, _ = eigs(data, k=2, which='LR')
            lambda2 = min(abs(i.real) for i in eigenvalues)

            # Find smallest real part
            eigenvalues, _ = eigs(data, k=1, which='SR')
            lambdan = eigenvalues[0].real
        else:
            eigenvals = sorted(np.linalg.eigvals(data.toarray()), reverse=True)
            lambda2 = eigenvals[1]
            lambdan = eigenvals[-1]

        beta = max(abs(lambda2), abs(lambdan))
        return data, beta


class TwoDimGridGraph(UndirectedGraph):
    def __init__(self, n):
        self.n = n ** 2
        self._data, self._beta = self._compute_beta(n)

    def _compute_beta(self, length):
        n = length ** 2

        def index_to_position(index):
            # return a two dim index; index starts from 0
            i = index // length
            j = index % length
            return i, j

        def position_to_index(i, j):
            return i * length + j

        def determine_neighborhood_index(index):
            i, j = index_to_position(index)
            if i > 0:
                yield position_to_index(i - 1, j)
            if i < length - 1:
                yield position_to_index(i+1, j)
            if j > 0:
                yield position_to_index(i, j - 1)
            if j < length - 1:
                yield position_to_index(i, j + 1)

        def degree(index):
            i, j = index_to_position(index)
            return sum([i > 0, i < length - 1, j > 0, j < length - 1])

        row = []
        col = []
        data = []
        for index in range(n):
            index_degree = degree(index)
            remaining_weight = 1
            for neighborhood_index in determine_neighborhood_index(index):
                neighborhood_index_degree = degree(index)

                row.append(index)
                col.append(neighborhood_index)

                weight = 1 / (1 + max(index_degree, neighborhood_index_degree))
                remaining_weight -= weight
                data.append(weight)
            else:
                row.append(index)
                col.append(index)
                data.append(remaining_weight)

        data = coo_matrix((data, (row, col)), shape=(n, n)).tocsr()

        if n > 3:
            # Find largest real part
            eigenvalues, _ = eigs(data, k=2, which='LR')
            lambda2 = min(abs(i.real) for i in eigenvalues)

            # Find smallest real part
            eigenvalues, _ = eigs(data, k=1, which='SR')
            lambdan = eigenvalues[0].real
        else:
            eigenvals = sorted(np.linalg.eigvals(data.toarray()), reverse=True)
            lambda2 = eigenvals[1]
            lambdan = eigenvals[-1]

        beta = max(abs(lambda2), abs(lambdan))
        return data, beta

    @property
    def n_edges(self):
        raise NotImplementedError

    @property
    def n_nodes(self):
        raise NotImplementedError

    @property
    def beta(self):
        return self._beta

    @property
    def matrix(self):
        return self._data

    def get_neighborhood(self, node_id):
        row = self._data.getrow(node_id)
        _, cols = row.nonzero()
        vals = row.data
        return {int(c): v for c, v in zip(cols, vals)}


class NConnectedCycleGraph(UndirectedGraph):
    def __init__(self, n, connectivity):
        assert (2 * connectivity + 1) < n
        # connectivity-cycle
        self.connectivity = connectivity
        # size of the ring
        self.n = n
        self._data, self._beta = self._compute_beta(n)

    def _compute_beta(self, n):

        weight = 1 / (self.connectivity * 2 + 1)
        row = []
        col = []
        data = []
        for index in range(n):
            for offset in range(1, self.connectivity + 1):
                row.append(index)
                col.append((index + offset) % n)
                data.append(weight)
                row.append(index)
                col.append((index - offset) % n)
                data.append(weight)
            else:
                row.append(index)
                col.append(index)
                data.append(weight)

        data = coo_matrix((data, (row, col)), shape=(n, n)).tocsr()

        if n > 3:
            # Find largest real part
            eigenvalues, _ = eigs(data, k=2, which='LR')
            lambda2 = min(abs(i.real) for i in eigenvalues)

            # Find smallest real part
            eigenvalues, _ = eigs(data, k=1, which='SR')
            lambdan = eigenvalues[0].real
        else:
            eigenvals = sorted(np.linalg.eigvals(data.toarray()), reverse=True)
            lambda2 = eigenvals[1]
            lambdan = eigenvals[-1]

        beta = max(abs(lambda2), abs(lambdan))
        return data, beta

    @property
    def n_edges(self):
        raise NotImplementedError

    @property
    def n_nodes(self):
        raise NotImplementedError

    @property
    def beta(self):
        return self._beta

    @property
    def matrix(self):
        return self._data

    def get_neighborhood(self, node_id):
        row = self._data.getrow(node_id)
        _, cols = row.nonzero()
        vals = row.data
        return {int(c): v for c, v in zip(cols, vals)}


class TimeVaryingCompleteGraph(CompleteGraph):
    def __init__(self, n, network_stability):
        self.n = n
        self._data = np.ones((self.n, self.n)) / self.n
        self.sequence = self.random_sequence()
        self.network_stability = network_stability
        assert network_stability >= 0.0 and network_stability <= 1.0

    def random_sequence(self):
        random.seed(2018)

        # generate a disappear rate for each node
        thresholds = [self.network_stability for i in range(self.n)]
        while True:
            visible = [random.uniform(0, 1) for i in range(self.n)]
            # Visible data
            visible = [
                1 if i < th else 0 for i, th in zip(visible, thresholds)]
            yield visible

    @property
    def n_edges(self):
        return self.n * (self.n - 1) / 2

    @property
    def n_nodes(self):
        return self.n

    @property
    def beta(self):
        return 0

    @property
    def matrix(self):
        return self._data

    def get_neighborhood(self, node_id):
        visible = self.sequence.__next__()
        pair = list(filter(lambda x: x[1] == 1, zip(range(self.n), visible)))
        d = len(pair)
        return {k: 1 / d for k, _ in pair}


def define_graph_topology(n_nodes, graph_topology,  **args):
    """Return the required graph object.

    Parameters
    ----------
    n_nodes : {int}
        Number of nodes in the network.
    graph_topology : {str}
        A string describing the graph topology

    Returns
    -------
    Graph
        A graph with specified information.
    """
    if graph_topology == 'ring':
        graph = RingGraph(n_nodes)
    elif graph_topology == 'complete':
        graph = CompleteGraph(n_nodes)
    elif graph_topology == 'time_varying_complete':
        graph = TimeVaryingCompleteGraph(n_nodes, args['network_stability'])
    elif graph_topology == 'non_uniform_weight_ring_graph':
        local_weight = args['local_weight']
        graph = NonUniformWeightRingGraph(n_nodes, local_weight)
    elif graph_topology == 'connected_cycle':
        n_connectivity = args['n_connectivity']
        graph = NConnectedCycleGraph(n_nodes, n_connectivity)
    elif graph_topology == 'grid':
        edge_length = int(np.sqrt(n_nodes))
        assert edge_length ** 2 == n_nodes
        graph = TwoDimGridGraph(edge_length)
    return graph
