from typing import List

import numpy as np
from scipy.stats import poisson

from opticverge.core.generator.options_generator import rand_options
from opticverge.core.globals import DEFAULT_SAMPLE_SIZE
from opticverge.core.numeric.safe import safe_value


def rand_int(
        min_val: int or np.int8 or np.int16 or np.int32 or np.int64 or np.int128,
        max_val: int or np.int8 or np.int16 or np.int32 or np.int64 or np.int128,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        rounding: int = None,
        output_dtype: int or np.int8 or np.int16 or np.int32 or np.int64 or np.int128 = None):
    """ Generates an int from a discrete uniform distribution between the minimum and maximum boundary

    Args:
        min_val (int, np.int8, np.int16, np.int32, np.int64, np.int128): The lower boundary.
        max_val (int, np.int8, np.int16, np.int32, np.int64, np.int128): The upper boundary.
        sample_size (int): Defaults to DEFAULT_SAMPLE_SIZE. The number of samples to take from the distribution
        rounding (int): The amount of rounding to apply
        output_dtype (int, np.int8, np.int16, np.int32, np.int64, np.int128): Defaults to np.int64.

    Returns:
        np.int64 by default or the output_dtype
    """

    if min_val is None or max_val is None:
        raise ValueError(
            "rand_int: min_val {} or max_val {} cannot be None".format(min_val, max_val)
        )

    options: List[np.int64] = np.random.randint(min_val, max_val, sample_size)

    choice: np.int64 = rand_options(options)

    return safe_value(choice, min_val, max_val, rounding, output_dtype)


def rand_poisson(value: int or np.int8 or np.int16 or np.int32 or np.int64 or np.int128 = None,
                 min_val: int or np.int8 or np.int16 or np.int32 or np.int64 or np.int128 = None,
                 max_val: int or np.int8 or np.int16 or np.int32 or np.int64 or np.int128 = None,
                 rounding: int = None,
                 sample_size: int = DEFAULT_SAMPLE_SIZE,
                 output_dtype: int or np.int8 or np.int16 or np.int32 or np.int64 or np.int128 = None) -> int or np.int8 or np.int16 or np.int32 or np.int64 or np.int128:
    """ Generates a random int from a poisson distribution

    Args:
        value (int, np.int8, np.int16, np.int32, np.int64, np.int128): The centre of the distribution
        min_val (int, np.int8, np.int16, np.int32, np.int64, np.int128): Defaults to None. The lower boundary.
        max_val (int, np.int8, np.int16, np.int32, np.int64, np.int128): Defaults to None. The upper boundary.
        rounding (int): Defaults to None. The number of places to round to.
        sample_size (int): Defaults to None. The number of samples to take from the distribution.
        output_dtype (int, np.int8, np.int16, np.int32, np.int64, np.int128): Defaults  to None. If set converts the output value to the dtype or uses the default of the output of the distribution

    Returns:
        Any: np.int64 by default, uses the output dtype if set
    """
    if value is None:
        if min_val is not None and max_val is not None:
            value = rand_int(min_val, max_val)

    options: List[np.int64] = poisson.rvs(value, size=sample_size)

    choice: np.int64 = rand_options(options)

    return safe_value(choice, min_val, max_val, rounding, output_dtype)
