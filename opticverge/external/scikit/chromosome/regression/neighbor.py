from collections import OrderedDict

import psutil
from sklearn.neighbors import KNeighborsRegressor

from opticverge.core.chromosome.class_chromosome import ClassChromosome
from opticverge.core.chromosome.distribution.int_distribution_chromosome import RandPoissonChromosome
from opticverge.core.chromosome.options_chromosome import RandOptionsChromosome
from opticverge.core.generator.int_distribution_generator import rand_int
from opticverge.core.globals import DEFAULT_NUM_JOBS


class KNeighborsRegressorChromosome(ClassChromosome):
    def __init__(self, num_jobs=None):

        super(KNeighborsRegressorChromosome, self).__init__(
            KNeighborsRegressor,
            self.genotype_factory(),
            OrderedDict({
                "n_jobs": num_jobs if num_jobs is not None else DEFAULT_NUM_JOBS
            })
        )

    def genotype_factory(self):
        return OrderedDict({
            "n_neighbors" : RandPoissonChromosome(
                value=4,
                min_val=2,
                max_val=None,
                output_dtype=int
            ),
            "weights": RandOptionsChromosome(options=[
                "uniform",
                "distance"
            ]),
            "algorithm": RandOptionsChromosome(options=[
                "ball_tree",
                "kd_tree",
                "brute"
            ]),
            "metric": RandOptionsChromosome(options=[
                "euclidean",
                "manhattan",
                "chebyshev"
            ]),
            "leaf_size": RandPoissonChromosome(
                value=rand_int(2, 32),
                min_val=3,
                max_val=None,
                output_dtype=int
            )
        })
