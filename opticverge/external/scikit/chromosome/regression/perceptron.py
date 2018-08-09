from collections import OrderedDict

from sklearn.neural_network import MLPRegressor

from opticverge.core.chromosome.array_chromosome import RandArrayChromosome
from opticverge.core.chromosome.class_chromosome import ClassChromosome
from opticverge.core.chromosome.distribution.bool_distribution_chromosome import RandUniformBooleanChromosome
from opticverge.core.chromosome.distribution.int_distribution_chromosome import RandPoissonChromosome
from opticverge.core.chromosome.options_chromosome import RandOptionsChromosome
from opticverge.core.generator.int_distribution_generator import rand_int
from opticverge.core.globals import INT32_MAX


class MLPRegressorChromosome(ClassChromosome):

    def __init__(self, layers: int = 6, min_layers: int = 2, max_layers=512):
        super(MLPRegressorChromosome, self).__init__(
            MLPRegressor,
            self.genotype_factory(layers, min_layers, max_layers)
        )

    def genotype_factory(self, layers: int = 6, min_layers: int = 2, max_layers=512):
        return OrderedDict({
            "hidden_layer_sizes": RandArrayChromosome(
                length=layers,
                fixed=False,
                generator=RandPoissonChromosome(
                    value=rand_int(min_val=min_layers, max_val=max_layers),
                    min_val=min_layers,
                    max_val=max_layers,
                    output_dtype=int
                )
            ),
            "activation": RandOptionsChromosome(
                options=[
                    "logistic",
                    "tanh",
                    "relu",
                    "identity"
                ]
            ),
            "solver": RandOptionsChromosome(
                options=[
                    "lbfgs",
                    "sgd",
                    "adam"
                ]
            ),
            "learning_rate": RandOptionsChromosome(
                options=[
                    "constant",
                    "invscaling",
                    "adaptive"
                ]
            ),
            "shuffle": RandUniformBooleanChromosome(),
            "random_state": RandPoissonChromosome(
                value=rand_int(1, INT32_MAX),
                min_val=1,
                max_val=INT32_MAX,
                rounding=None,
                output_dtype=int
            ),
            "warm_start": RandUniformBooleanChromosome(),
            "max_iter": RandOptionsChromosome(
                options=[
                    256,
                    512,
                    1024,
                    2048,
                    4096
                ]
            ),
        })

