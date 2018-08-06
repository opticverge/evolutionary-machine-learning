from typing import Any

import numpy as np

from opticverge.core.generator.options_generator import rand_options
from opticverge.core.numeric.safe import safe_value


def rand_real(min_val: float or np.float32 or np.float64 = 0.0,
              max_val: float or np.float32 or np.float64 = 1.0,
              sample_size: int = 1,
              rounding: int = None,
              output_dtype: Any = None):
    """ Generates a random number between the min_val and max_val using the normal distribution

    Args:
        min_val (float, np.float32, np.float64, optional): Defaults to 0.0. The lower boundary
        max_val (float, np.float32, np.float64, optional): Defaults to 1.0. The upper boundary
        sample_size(int): Defaults to 1. The number of samples to take from the normal distribution
        rounding (int, optional): Defaults to None. If set applies np.round to the value
        output_dtype (Any, optional): Defaults to None. If set converts the value to the desired type
    
    Raises:
        ValueError: 
    
    Returns:
        Any: np.float64 by default otherwise the type specified in output_dtype
    """
    if min_val is None or max_val is None:
        raise ValueError(
            "rand_real: min_val {} or max_val {} cannot be None".format(min_val, max_val)
        )

    if output_dtype is None:
        output_dtype = np.float64

    options = np.random.uniform(min_val, max_val, sample_size)

    choice = rand_options(options)

    return safe_value(choice, min_val, max_val, rounding, output_dtype)
