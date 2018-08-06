from collections import OrderedDict

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from opticverge.core.chromosome.class_chromosome import ClassChromosome
from opticverge.core.chromosome.distribution.bool_distribution_chromosome import RandUniformBooleanChromosome
from opticverge.core.chromosome.distribution.int_distribution_chromosome import RandPoissonChromosome
from opticverge.core.chromosome.distribution.real_distribution_chromosome import RandGaussChromosome
from opticverge.core.chromosome.options_chromosome import RandOptionsChromosome
from opticverge.core.generator.int_distribution_generator import rand_int
from opticverge.core.globals import INT32_MAX


class GradientBoostingRegressorChromosome(ClassChromosome):
    """ The chromosome class for the GradientBoostingRegressor from Scikit-learn """

    def __init__(self, n_estimators: int or None = None, learning_rate: float or None = None,
                 max_depth: int or None = None):
        """ The constructor for this class

        Args:
            n_estimators: The number of estimators to start from, defaults to 128
            learning_rate: The learning rate for the regressor, defaults to 0.1
            max_depth: The maximum depth of the tree, defaults to 6
        """
        super(GradientBoostingRegressorChromosome, self).__init__(
            GradientBoostingRegressor,
            self.blueprint_factory(n_estimators, learning_rate, max_depth)
        )

    def blueprint_factory(self, n_estimators=None, learning_rate=None, max_depth=None, alpha=None):
        return OrderedDict({
            "n_estimators": RandPoissonChromosome(
                value=n_estimators if n_estimators is not None else 128,
                min_val=2
            ),
            "loss": RandOptionsChromosome(
                options=[
                    "ls",
                    "lad",
                    "huber",
                    "quantile"
                ]
            ),
            "learning_rate": RandGaussChromosome(
                value=learning_rate if learning_rate is not None else 0.1,
                min_val=0.001,
                max_val=0.999,
                rounding=3,
                output_dtype=np.float64
            ),
            "max_features": RandOptionsChromosome(
                options=[
                    "auto",
                    "sqrt",
                    "log2"
                ]
            ),
            "max_depth": RandPoissonChromosome(
                value=max_depth if max_depth is not None else 6,
                min_val=2,
                max_val=None,
                rounding=None,
                output_dtype=np.int64
            ),
            "criterion": RandOptionsChromosome(
                options=[
                    "mse",
                    "mae",
                    "friedman_mse"
                ]
            ),
            "warm_start": RandUniformBooleanChromosome(),
            "alpha": RandGaussChromosome(
                value=alpha if alpha is not None else 0.9,
                min_val=0.01,
                max_val=0.99,
                rounding=2,
                output_dtype=np.float64
            ),
            "random_state": RandPoissonChromosome(
                value=rand_int(1, INT32_MAX),
                min_val=1,
                max_val=INT32_MAX,
                rounding=None,
                output_dtype=int
            )
        })
