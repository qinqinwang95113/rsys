import numpy as np
import random
import warnings
from collections import defaultdict, deque
from abc import ABC, abstractmethod
from recsys_framework.graph.graph_utils import is_real_iterable
from collections import Counter
from networkx import Graph as nxGraph
from tqdm import tqdm

__all__ = [
    "UniformRandomWalker",
    "BiasedRandomWalk",
    #"UniformRandomMetaPathWalk",
    #"SampledBreadthFirstWalk",
    #"SampledHeterogeneousBreadthFirstWalk",
    #"DirectedBreadthFirstNeighbours",
]


class BaseWalker(ABC):
    """
    Abstract class to explore graph through walks
    """

    def __init__(self, graph, seed=None):
        self.graph = graph

        # Initialize the random state
        self._check_seed(seed)
        self._random_state = random.Random(seed)

        # Initialize a numpy random state (for numpy random methods)
        self._np_random_state = np.random.RandomState(seed=seed)

        # Store the last rw request
        self.last_walk = None

        #TODO: has to be changed when the graph type will be changed
        # We require a NetworkX graph for this
        if not isinstance(graph, nxGraph):
            raise TypeError("graph must be a NetworkX graph")

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        To be overridden by subclasses. It is the main entry point for performing random walks on the given
        graph.
        It should return the sequences of nodes in each random walk.
        """
        pass

    def run_count(self, *args, **kwargs):
        """
        Return a list of dictionary
        [{node_id: count,...}, ...]
        obtained from run method
        """

        if self.last_walk is None:
            raise EnvironmentError("No Last walk saved! \n to perform and store a walk use the \"run\" method.")

        return [Counter(run) for run in tqdm(self.run(*args, **kwargs))]


    #########################################################################################################
    ##########                                                                                     ##########
    ##########                                UTILS METHODS                                        ##########
    ##########                                                                                     ##########
    #########################################################################################################

    def _check_seed(self, seed):
        if seed is not None:
            if type(seed) != int:
                self._raise_error(
                    "The random number generator seed value, seed, should be integer type or None."
                )
            if seed < 0:
                self._raise_error(
                    "The random number generator seed value, seed, should be non-negative integer or None."
                )

    def _get_random_state(self, seed):
        """
        Args:
            seed: The optional seed value for a given run.
        Returns:
            The random state as determined by the seed.
        """
        if seed is None:
            # Restore the random state
            return self._random_state
        # seed the random number generator
        return random.Random(seed)

    def _raise_error(self, msg):
        raise ValueError("({}) {}".format(type(self).__name__, msg))

    def _check_common_parameters(self, nodes, n, length, seed):
        """
        Checks that the parameter values are valid or raises ValueError exceptions with a message indicating the
        parameter (the first one encountered in the checks) with invalid value.
        Args:
            nodes: <list> A list of root node ids from which to commence the random walks.
            n: <int> Number of walks per node id.
            length: <int> Maximum length of each walk.
            seed: <int> Random number generator seed.
        """
        self._check_nodes(nodes)
        self._check_repetitions(n)
        self._check_length(length)
        self._check_seed(seed)

    def _check_nodes(self, nodes):
        if nodes is None:
            self._raise_error("A list of root node IDs was not provided.")
        if not is_real_iterable(nodes):
            self._raise_error("Nodes parameter should be an iterable of node IDs.")
        if (
            len(nodes) == 0
        ):  # this is not an error but maybe a warning should be printed to inform the caller
            warnings.warn(
                "No root node IDs given. An empty list will be returned as a result.",
                RuntimeWarning,
                stacklevel=3,
            )

    def _check_repetitions(self, n):
        if type(n) != int:
            self._raise_error(
                "The number of walks per root node, n, should be integer type."
            )
        if n <= 0:
            self._raise_error(
                "The number of walks per root node, n, should be a positive integer."
            )

    def _check_length(self, length):
        if type(length) != int:
            self._raise_error("The walk length, length, should be integer type.")
        if length <= 0:
            # Technically, length 0 should be okay, but by consensus is invalid.
            self._raise_error("The walk length, length, should be a positive integer.")

    def _check_sizes(self, n_size):
        err_msg = "The neighbourhood size must be a list of non-negative integers."
        if not isinstance(n_size, list):
            self._raise_error(err_msg)
        if len(n_size) == 0:
            # Technically, length 0 should be okay, but by consensus it is invalid.
            self._raise_error("The neighbourhood size list should not be empty.")
        for d in n_size:
            if type(d) != int or d < 0:
                self._raise_error(err_msg)


class UniformRandomWalker(BaseWalker):
    """
    Performs uniform random walks on the given graph
    """

    def run(self, nodes, n, length, seed=None):
        """
        Perform a random walk starting from the root nodes.
        Args:
            nodes (list): The root nodes as a list of node IDs
            n (int): Total number of random walks per root node
            length (int): Maximum length of each random walk
            seed (int, optional): Random number generator seed; default is None
        Returns:
            List of lists of nodes ids for each of the random walks
        """
        self._check_common_parameters(nodes, n, length, seed)
        rnd_seed = self._get_random_state(seed)

        # for each root node, do n walks
        self.last_walk = [self._walk(rnd_seed, node, length) for node in nodes for _ in range(n)]
        return self.last_walk

    def _walk(self, rnd_seed, start_node, length):
        walk = [start_node]
        current_node = start_node
        for _ in range(length - 1):
            neighbours = list(self.graph.neighbors(current_node))
            if not neighbours:
                # dead end, so stop
                break
            else:
                # has neighbours, so pick one to walk to
                current_node = rnd_seed.choice(neighbours)
            walk.append(current_node)
        return walk


def naive_weighted_choices(rs, weights):
    """
    Select an index at random, weighted by the iterator `weights` of
    arbitrary (non-negative) floats. That is, `x` will be returned
    with probability `weights[x]/sum(weights)`.
    For doing a single sample with arbitrary weights, this is much (5x
    or more) faster than numpy.random.choice, because the latter
    requires a lot of preprocessing (normalized probabilties), and
    does a lot of conversions/checks/preprocessing internally.
    """

    # divide the interval [0, sum(weights)) into len(weights)
    # subintervals [x_i, x_{i+1}), where the width x_{i+1} - x_i ==
    # weights[i]
    subinterval_ends = []
    running_total = 0
    for w in weights:
        if w < 0:
            raise ValueError("Detected negative weight: {}".format(w))
        running_total += w
        subinterval_ends.append(running_total)

    # pick a place in the overall interval
    x = rs.random() * running_total

    # find the subinterval that contains the place, by looking for the
    # first subinterval where the end is (strictly) after it
    for idx, end in enumerate(subinterval_ends):
        if x < end:
            break

    return idx


class BiasedRandomWalk(BaseWalker):
    """
    Performs biased second order random walks (like those used in Node2Vec algorithm
    https://snap.stanford.edu/node2vec/) controlled by the values of two parameters p and q.
    """

    def run(self, nodes, n, length, p=1.0, q=1.0, seed=None, weighted=False):

        """
        Perform a random walk starting from the root nodes.
        Args:
            nodes (list): The root nodes as a list of node IDs
            n (int): Total number of random walks per root node
            length (int): Maximum length of each random walk
            p (float, default 1.0): Defines probability, 1/p, of returning to source node
            q (float, default 1.0): Defines probability, 1/q, for moving to a node away from the source node
            seed (int, optional): Random number generator seed; default is None
            weighted (bool, default False): Indicates whether the walk is unweighted or weighted
        Returns:
            List of lists of nodes ids for each of the random walks
        """
        self._check_common_parameters(nodes, n, length, seed)
        self._check_weights(p, q, weighted)
        rs = self._get_random_state(seed)

        if weighted:
            # Check that all edge weights are greater than or equal to 0.
            # Also, if the given graph is a MultiGraph, then check that there are no two edges between
            # the same two nodes with different weights.
            for node in self.graph.nodes():
                # TODO Encapsulate edge weights
                for neighbor in self.graph.neighbors(node):

                    wts = set()
                    for weight in self.graph._edge_weights(node, neighbor):
                        if weight is None or np.isnan(weight) or weight == np.inf:
                            self._raise_error(
                                "Missing or invalid edge weight ({}) between ({}) and ({}).".format(
                                    weight, node, neighbor
                                )
                            )
                        if not isinstance(weight, (int, float)):
                            self._raise_error(
                                "Edge weight between nodes ({}) and ({}) is not numeric ({}).".format(
                                    node, neighbor, weight
                                )
                            )
                        if weight < 0:  # check if edge has a negative weight
                            self._raise_error(
                                "An edge weight between nodes ({}) and ({}) is negative ({}).".format(
                                    node, neighbor, weight
                                )
                            )

                        wts.add(weight)
                    if len(wts) > 1:
                        # multigraph with different weights on edges between same pair of nodes
                        self._raise_error(
                            "({}) and ({}) have multiple edges with weights ({}). Ambiguous to choose an edge for the random walk.".format(
                                node, neighbor, list(wts)
                            )
                        )

        ip = 1.0 / p
        iq = 1.0 / q

        walks = []
        for node in nodes:  # iterate over root nodes
            for walk_number in range(n):  # generate n walks per root node
                # the walk starts at the root
                walk = [node]

                neighbours = list(self.graph.neighbors(node))

                previous_node = node
                previous_node_neighbours = neighbours

                # calculate the appropriate unnormalised transition
                # probability, given the history of the walk
                def transition_probability(nn, current_node, weighted):

                    if weighted:
                        # TODO Encapsulate edge weights
                        weight_cn = self.graph._edge_weights(current_node, nn)[0]
                    else:
                        weight_cn = 1.0

                    if nn == previous_node:  # d_tx = 0
                        return ip * weight_cn
                    elif nn in previous_node_neighbours:  # d_tx = 1
                        return 1.0 * weight_cn
                    else:  # d_tx = 2
                        return iq * weight_cn

                if neighbours:
                    current_node = rs.choice(neighbours)
                    for _ in range(length - 1):
                        walk.append(current_node)
                        neighbours = list(self.graph.neighbors(current_node))

                        if not neighbours:
                            break

                        # select one of the neighbours using the
                        # appropriate transition probabilities
                        choice = naive_weighted_choices(
                            rs,
                            (
                                transition_probability(nn, current_node, weighted)
                                for nn in neighbours
                            ),
                        )

                        previous_node = current_node
                        previous_node_neighbours = neighbours
                        current_node = neighbours[choice]

                walks.append(walk)

        return walks

    def _check_weights(self, p, q, weighted):
        """
        Checks that the parameter values are valid or raises ValueError exceptions with a message indicating the
        parameter (the first one encountered in the checks) with invalid value.
        Args:
            p: <float> The backward walk 'penalty' factor.
            q: <float> The forward walk 'penalty' factor.
            weighted: <False or True> Indicates whether the walk is unweighted or weighted.
       """
        if p <= 0.0:
            self._raise_error("Parameter p should be greater than 0.")

        if q <= 0.0:
            self._raise_error("Parameter q should be greater than 0.")

        if type(weighted) != bool:
            self._raise_error(
                "Parameter weighted has to be either False (unweighted random walks) or True (weighted random walks)."
            )
