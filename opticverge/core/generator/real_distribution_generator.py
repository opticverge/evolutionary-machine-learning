import numpy as np
from scipy.stats import norm

from opticverge.core.generator.options_generator import rand_options
from opticverge.core.globals import DEFAULT_SAMPLE_SIZE
from opticverge.core.numeric.safe import safe_value


def rand_gauss(
        value: float or np.float32 or np.float64,
        min_val: float or np.float32 or np.float64 = None,
        max_val: float or np.float32 or np.float64 = None,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        rounding: int = None,
        output_dtype: float or np.float32 or np.float64 = None) -> float or np.float32 or np.float64:
    """ Generates a random float from a normal distribution

    Args:
        value: The centre of the distribution
        min_val: The lower boundary
        max_val: The upper boundary
        sample_size: The number of samples generate
        rounding: The amount of rounding to apply to the samples
        output_dtype: The output type of the operation

    Returns:
        float, np.float32, np.float64
    """

    # retrieve the options from sampling the normal distribution
    choices = norm.rvs(size=sample_size)

    # select one of them
    choice = rand_options(choices)

    # change the value based on the choice
    result = (choice * np.float64(value)) + np.float64(value)

    # safely evaluate the result respecting the boundaries, rounding an output type
    return safe_value(result, min_val, max_val, rounding, output_dtype)
