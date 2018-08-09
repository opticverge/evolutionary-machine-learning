from collections import OrderedDict

from sklearn.tree import DecisionTreeRegressor

from opticverge.core.chromosome.class_chromosome import ClassChromosome
from opticverge.core.chromosome.distribution.int_distribution_chromosome import RandPoissonChromosome
from opticverge.core.chromosome.options_chromosome import RandOptionsChromosome
from opticverge.core.generator.int_distribution_generator import rand_int
from opticverge.core.globals import INT32_MAX


class DecisionTreeRegressorChromosome(ClassChromosome):
    def __init__(self, max_depth: int = None):
        super(DecisionTreeRegressorChromosome, self).__init__(
            DecisionTreeRegressor,
            self.genotype_factory(max_depth)
        )

    def genotype_factory(self, max_depth: int = None):
        return OrderedDict({
            "splitter": RandOptionsChromosome(
                options=[
                    "best",
                    "random"
                ]
            ),
            "criterion": RandOptionsChromosome(
                options=[
                    "mse",
                    "mae",
                    "friedman_mse"
                ]
            ),
            "max_depth": RandPoissonChromosome(
                value=max_depth if max_depth is not None else rand_int(2, 16),
                min_val=2,
                max_val=None,
                rounding=None,
                output_dtype=int
            ),
            "max_features": RandOptionsChromosome(
                options=[
                    "auto",
                    "sqrt",
                    "log2"
                ]
            ),
            "random_state": RandPoissonChromosome(
                value=rand_int(1, INT32_MAX),
                min_val=1,
                max_val=INT32_MAX,
                rounding=None,
                output_dtype=int
            )
        })
