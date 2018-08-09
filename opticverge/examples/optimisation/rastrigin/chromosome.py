import numpy as np

from opticverge.core.chromosome.array_chromosome import RandArrayChromosome
from opticverge.core.chromosome.distribution.real_distribution_chromosome import RandGaussChromosome
from opticverge.core.generator.real_generator import rand_real


class RastriginChromosome(RandArrayChromosome):
    def __init__(self, dimensions: int):
        super(RastriginChromosome, self).__init__(
            length=dimensions,
            fixed=True,
            generator=RandGaussChromosome(
                value=rand_real(
                    min_val=-5.12,
                    max_val=5.12
                ),
                min_val=-5.12,
                max_val=5.12,
                rounding=2,
                output_dtype=np.float64
            )
        )
