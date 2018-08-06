from collections import OrderedDict

import numpy as np

from opticverge.core.chromosome.abstract_chromosome import AbstractChromosome
from opticverge.core.enum.objective import Objective
from opticverge.core.log.logger import DATA, data_logger
from opticverge.core.problem.abstract_problem import AbstractProblem
from opticverge.core.solver.abstract_solver import AbstractSolver


class OneMaxProblem(AbstractProblem):
    def __init__(self):
        super(OneMaxProblem, self).__init__(Objective.Maximisation, "One Max Problem")

    def objective_function(self, chromosome: AbstractChromosome):
        super(OneMaxProblem, self).objective_function(chromosome)
        chromosome.fitness = np.sum(chromosome.phenotype)

    def log_chromosome(self, chromosome: AbstractChromosome, solver: AbstractSolver):
        data_str = super(OneMaxProblem, self).log_chromosome(
            chromosome=chromosome,
            solver=solver,
            additional_data=OrderedDict({
                "phenotype": chromosome.phenotype
            })
        )
        data_logger.log(DATA, data_str)
