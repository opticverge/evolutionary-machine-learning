from typing import Dict

import numpy as np

from opticverge.core.chromosome.abstract_chromosome import AbstractChromosome
from opticverge.core.chromosome.array_chromosome import RandArrayChromosome
from opticverge.core.chromosome.distribution.int_distribution_chromosome import RandIntChromosome


class OneMaxChromosome(RandArrayChromosome):
    def blueprint_factory(self, **kwargs) -> Dict[str, AbstractChromosome]:
        pass

    def __init__(self, dimensions: int):
        super(OneMaxChromosome, self).__init__(
            length=dimensions,
            fixed=True,
            generator=RandIntChromosome(
                min_val=0,
                max_val=2,
                rounding=None,
                output_dtype=np.int8
            )
        )
