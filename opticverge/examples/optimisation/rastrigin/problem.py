from typing import Any, Dict

import numpy as np

from opticverge.core.chromosome.abstract_chromosome import AbstractChromosome
from opticverge.core.enum.objective import Objective
from opticverge.core.log.logger import data_logger, DATA
from opticverge.core.problem.abstract_problem import AbstractProblem
from opticverge.core.solver.abstract_solver import AbstractSolver


class RastriginProblem(AbstractProblem):
    def __init__(self):
        super(RastriginProblem, self).__init__(
            Objective.Minimisation,
            "Rastrigin"
        )

    def log_chromosome(self, chromosome: AbstractChromosome, solver: AbstractSolver,
                       additional_data: Dict[str, Any] = None, separator="|") -> str:
        data_str = super(RastriginProblem, self).log_chromosome(
            chromosome=chromosome,
            solver=solver
        )

        data_logger.log(DATA, data_str)

    def objective_function(self, chromosome: AbstractChromosome):
        x = chromosome.genotype
        d = len(x)
        formula_sum = 0.0
        for i, val in enumerate(x):
            formula_sum += np.square(val) - (10 * np.cos(2 * np.pi * val))

        chromosome.fitness = np.abs((10 * d) + formula_sum)
