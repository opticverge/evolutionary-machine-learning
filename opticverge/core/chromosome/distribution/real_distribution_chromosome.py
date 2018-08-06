from collections import OrderedDict

from opticverge.core.chromosome.function_chromosome import FunctionChromosome
from opticverge.core.generator.real_distribution_generator import rand_gauss
from opticverge.core.globals import DEFAULT_SAMPLE_SIZE


class RandGaussChromosome(FunctionChromosome):
    def __init__(self,
                 value,
                 min_val=None,
                 max_val=None,
                 rounding=None,
                 sample_size=DEFAULT_SAMPLE_SIZE,
                 output_dtype=None):
        super(RandGaussChromosome, self).__init__(
            rand_gauss,
            OrderedDict({
                "value": value,
                "min_val": min_val,
                "max_val": max_val,
                "rounding": rounding,
                "sample_size": sample_size,
                "output_dtype": output_dtype
            })
        )
