from collections import OrderedDict
from typing import Dict

from opticverge.core.chromosome.abstract_chromosome import AbstractChromosomeEntity, AbstractChromosome
from opticverge.core.chromosome.function_chromosome import FunctionChromosome
from opticverge.core.generator.int_distribution_generator import rand_int, rand_poisson
from opticverge.core.globals import DEFAULT_SAMPLE_SIZE


class RandIntChromosome(FunctionChromosome):

    def __init__(self,
                 min_val,
                 max_val,
                 rounding=None,
                 sample_size=DEFAULT_SAMPLE_SIZE,
                 output_dtype=None):
        super(RandIntChromosome, self).__init__(
            rand_int,
            OrderedDict({
                "min_val": min_val,
                "max_val": max_val,
                "rounding": rounding,
                "sample_size": sample_size,
                "output_dtype": output_dtype
            })
        )


class RandPoissonChromosome(FunctionChromosome):

    def blueprint_factory(self, value, min_val, max_val, rounding, sample_size, output_dtype) -> Dict[str, AbstractChromosome]:
        return OrderedDict({
            "value": value,
            "min_val": min_val,
            "max_val": max_val,
            "rounding": rounding,
            "sample_size": sample_size,
            "output_dtype": output_dtype
        })

    def __init__(self,
                 value,
                 min_val=None,
                 max_val=None,
                 rounding=None,
                 sample_size=DEFAULT_SAMPLE_SIZE,
                 output_dtype=None):
        super(RandPoissonChromosome, self).__init__(
            rand_poisson,
            self.blueprint_factory(
                value=value,
                min_val=min_val,
                max_val=max_val,
                rounding=rounding,
                sample_size=sample_size,
                output_dtype=output_dtype)
        )
