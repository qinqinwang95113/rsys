import numpy as np
import random
import warnings
from collections import defaultdict, deque
from abc import ABC, abstractmethod
from recsys_framework.graph.graph_utils import is_real_iterable

__all__ = [
    "UniformRandomWalk",
    "BiasedRandomWalk",
    "UniformRandomMetaPathWalk",
    "SampledBreadthFirstWalk",
    "SampledHeterogeneousBreadthFirstWalk",
    "DirectedBreadthFirstNeighbours",
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

        #TODO: change this when the GRAPH class will be implemented
        # We require a StellarGraph for this
        #if not isinstance(graph, StellarGraph):
        #    raise TypeError("graph must be a StellarGraph or StellarDiGraph.")

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        To be overridden by subclasses. It is the main entry point for performing random walks on the given
        graph.
        It should return the sequences of nodes in each random walk.
        """
        pass

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
        return [self._walk(rnd_seed, node, length) for node in nodes for _ in range(n)]

    def _walk(self, rnd_seed, start_node, length):
        walk = [start_node]
        current_node = start_node
        for _ in range(length - 1):
            neighbours = self.graph.neighbors(current_node)
            if not neighbours:
                # dead end, so stop
                break
            else:
                # has neighbours, so pick one to walk to
                current_node = rnd_seed.choice(neighbours)
            walk.append(current_node)

        return walk
