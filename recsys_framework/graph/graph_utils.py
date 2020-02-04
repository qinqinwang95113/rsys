import collections
import scipy.sparse as sp
from scipy.sparse.linalg import ArpackNoConvergence, eigsh
import numpy as np


def is_real_iterable(x):
    """
    Tests if x is an iterable and is not a string.
    Args:
        x: a variable to check for whether it is an iterable
    Returns:
        True if x is an iterable (but not a string) and False otherwise
    """
    return isinstance(x, collections.Iterable) and not isinstance(x, (str, bytes))