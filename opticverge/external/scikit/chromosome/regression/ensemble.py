from collections import OrderedDict

import numpy as np
import psutil
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from xgboost import XGBRegressor

from opticverge.core.chromosome.class_chromosome import ClassChromosome
from opticverge.core.chromosome.distribution.bool_distribution_chromosome import RandUniformBooleanChromosome
from opticverge.core.chromosome.distribution.int_distribution_chromosome import RandPoissonChromosome
from opticverge.core.chromosome.distribution.real_distribution_chromosome import RandGaussChromosome
from opticverge.core.chromosome.options_chromosome import RandOptionsChromosome
from opticverge.core.generator.int_distribution_generator import rand_int
from opticverge.core.generator.real_generator import rand_real
from opticverge.core.globals import INT32_MAX
from opticverge.external.scikit.chromosome.regression.tree import DecisionTreeRegressorChromosome


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

    def blueprint_factory(self,
                          n_estimators: int = None,
                          learning_rate: float = None,
                          max_depth: int = None,
                          alpha: float = None):
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


class XGBRegressorChromosome(ClassChromosome):
    def __init__(self,
                 max_depth: int = None,
                 learning_rate: float = None,
                 n_estimators: int = None,
                 num_jobs: int = None):
        super(XGBRegressorChromosome, self).__init__(
            XGBRegressor,
            self.blueprint_factory(
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators
            ),
            OrderedDict({
                "n_jobs": num_jobs if num_jobs is not None else psutil.cpu_count()
            })
        )

    def blueprint_factory(self,
                          max_depth: int = None,
                          learning_rate: float = None,
                          n_estimators: int = None):
        return OrderedDict({
            "max_depth": RandPoissonChromosome(
                value=max_depth if max_depth is not None else rand_int(2, 16),
                min_val=2,
                max_val=None,
                rounding=None,
                output_dtype=int
            ),
            "learning_rate": RandGaussChromosome(
                value=learning_rate if learning_rate is not None else 0.1,
                min_val=0.01,
                max_val=0.99,
                rounding=2
            ),
            "n_estimators": RandPoissonChromosome(
                value=n_estimators if n_estimators is not None else rand_int(2, 128),
                min_val=2,
                max_val=None,
                output_dtype=int
            ),
            "objective": RandOptionsChromosome(
                options=[
                    "reg:linear",
                    "reg:gamma",
                    "reg:tweedie"

                ]
            ),
            "booster": RandOptionsChromosome(
                options=[
                    "gbtree",
                    "gblinear",
                    "dart"
                ]
            ),
            "base_score": RandGaussChromosome(
                value=0.5,
                min_val=0.01,
                max_val=0.99,
                rounding=3
            ),
            "random_state": RandPoissonChromosome(
                value=rand_int(1, INT32_MAX),
                min_val=1,
                max_val=INT32_MAX,
                rounding=None,
                output_dtype=int
            ),
            "gamma": RandGaussChromosome(
                value=rand_real(),
                min_val=0.01,
                max_val=0.99,
                rounding=3
            )
        })


class AdaBoostRegressorChromosome(ClassChromosome):
    def __init__(self, n_estimators: int = None, learning_rate: float = None, regressor_chromosome=None):
        super(AdaBoostRegressorChromosome, self).__init__(
            AdaBoostRegressor,
            self.genotype_factory(n_estimators, learning_rate, regressor_chromosome)
        )

    def genotype_factory(self,
                         n_estimators: int = None,
                         learning_rate: float = None,
                         regressor_chromosome=None):
        return OrderedDict({
            "base_estimator": regressor_chromosome if regressor_chromosome is not None else DecisionTreeRegressorChromosome(),
            "n_estimators": RandPoissonChromosome(
                value=n_estimators if n_estimators is not None else 50,
                min_val=2,
                max_val=None,
                output_dtype=int
            ),
            "learning_rate": RandGaussChromosome(
                value=learning_rate if learning_rate is not None else rand_real(),
                min_val=0.01,
                max_val=1.0,
                rounding=3,
                output_dtype=np.float32
            ),
            "loss": RandOptionsChromosome(
                options=[
                    "linear",
                    "square",
                    "exponential"
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


class RandomForestRegressorChromosome(ClassChromosome):
    def __init__(self, max_depth: int = None, n_estimators: int = None, num_jobs=None):
        super(RandomForestRegressorChromosome, self).__init__(
            RandomForestRegressor,
            self.genotype_factory(max_depth, n_estimators),
            OrderedDict({
                "n_jobs": num_jobs if num_jobs is not None else int(psutil.cpu_count(logical=True) / 2)
            })
        )

    def genotype_factory(self, max_depth: int = None, n_estimators: int = None):
        return OrderedDict({
            "max_depth": RandPoissonChromosome(
                value=max_depth if max_depth is not None else 3,
                min_val=2,
                max_val=None,
                output_dtype=int
            ),
            "n_estimators": RandPoissonChromosome(
                value=n_estimators if n_estimators is not None else 10,
                min_val=2,
                max_val=None,
                output_dtype=int
            ),
            "criterion": RandOptionsChromosome(
                options=[
                    "mse",
                    "mae"
                ]
            ),
            "max_features": RandOptionsChromosome(
                options=[
                    "auto",
                    "sqrt",
                    "log2"
                ]
            ),
            "warm_start": RandUniformBooleanChromosome(),
            "bootstrap": RandUniformBooleanChromosome(),
            "random_state": RandPoissonChromosome(
                value=rand_int(1, INT32_MAX),
                min_val=1,
                max_val=INT32_MAX,
                rounding=None,
                output_dtype=int
            )
        })
