from typing import List, Any

import numpy as np


def rand_options(options: List[Any] or int, size: int = None, replacement: bool = False):
    """

    Args:
        options: The list of items to choose from
        size: The number of items to choose
        replacement: Select unique (false) or repeating (true) values

    Returns:
        Any, List[Any]

    """
    return np.random.choice(options, size, replacement)
