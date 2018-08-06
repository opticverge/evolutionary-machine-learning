from typing import Any

import numpy as np


def safe_value(
        value: int or float,
        min_val: int or float or None,
        max_val: int or float or None,
        rounding: int,
        output_dtype: int or float or None):
    """ Safely ensures that the value exists within the bounds of the
    minimum and maximum if specified.

    Args:
        value: The value to be modified
        min_val: The lower boundary for the value
        max_val: The upper boundary for the value
        rounding: The amount of rounding to apply
        output_dtype: The output type of the value

    Returns:

    """

    if min_val is not None and max_val is not None:

        if value < min_val or value > max_val:

            while value < min_val or value > max_val:

                if value < min_val:
                    value = max_val - abs(min_val - value)

                if value > max_val:
                    value = min_val + abs(value - max_val)

    elif min_val is not None and max_val is None:
        value = abs(min_val - value) + min_val

    elif min_val is None and max_val is not None:
        value = max_val - abs(value - max_val)

    value = safe_round(value, rounding)

    if output_dtype is not None:
        value = output_dtype(value)

    return value


def safe_round(value: Any, rounding: int):
    if rounding:
        return np.round(value, rounding)

    return value
