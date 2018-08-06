def scale(value: int or float, origin_lower: int or float, origin_upper, scale_lower, scale_upper):
    """ Translate a value from one numerical range to another

    Args:
        value(int,float): The current value
        origin_lower(int,float): The lower boundary of the original domain
        origin_upper(int,float): The upper boundary of the original domain
        scale_lower(int,float): The lower boundary of the new domain
        scale_upper(int,float): The upper boundary of the new domain

    Returns:
        int, float
    """

    return scale_lower + (value - origin_lower) * ((scale_upper - scale_lower) / (origin_upper - origin_lower))
