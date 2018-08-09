from typing import Any, Dict, List

import numpy as np

from opticverge.core.chromosome.abstract_chromosome import AbstractChromosome
from opticverge.core.enum.objective import Objective
from opticverge.core.log.logger import data_logger, DATA
from opticverge.core.problem.abstract_problem import AbstractProblem
from opticverge.core.solver.abstract_solver import AbstractSolver


class AckleyProblem(AbstractProblem):
    def __init__(self):
        super(AckleyProblem, self).__init__(
            Objective.Minimisation,
            "Ackley"
        )

    def log_chromosome(self, chromosome: AbstractChromosome, solver: AbstractSolver,
                       additional_data: Dict[str, Any] = None, separator="|") -> str:
        data_str = super(AckleyProblem, self).log_chromosome(
            chromosome=chromosome,
            solver=solver
        )

        data_logger.log(DATA, data_str)

    def objective_function(self, chromosome: AbstractChromosome):
        a = 20
        b = 0.2
        c = 2 * np.pi
        d = len(chromosome.phenotype)
        x: List[float] = chromosome.phenotype
        sum_squared = np.sum(np.square(x))
        sum_cos = 0.0
        for i, val in enumerate(x):
            sum_cos += c * np.cos(val)

        chromosome.fitness = np.abs(
            -a * np.exp(-b * np.sqrt(1 / d * sum_squared)) - np.exp(1 / d * sum_cos) + a + np.exp(1))
