import numpy as np


def rand_uniform_bool(loc=0.5) -> bool:
    """ Generates a boolean derived from a uniform distribution biased towards the value of loc

    Args:
        loc: The bias, if higher tends towards producing false more and vice versa

    Returns:
        bool
    """
    return np.random.uniform() > loc


