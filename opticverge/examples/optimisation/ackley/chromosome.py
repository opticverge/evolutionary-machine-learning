import numpy as np

from opticverge.core.chromosome.array_chromosome import RandArrayChromosome
from opticverge.core.chromosome.distribution.real_distribution_chromosome import RandGaussChromosome
from opticverge.core.generator.real_generator import rand_real


class AckleyChromosome(RandArrayChromosome):
    def __init__(self, dimensions: int):
        super(AckleyChromosome, self).__init__(
            length=dimensions,
            fixed=True,
            generator=RandGaussChromosome(
                value=rand_real(
                    min_val=-32.768,
                    max_val=32.768
                ),
                min_val=-32.768,
                max_val=32.768,
                rounding=3,
                output_dtype=np.float64
            )
        )
