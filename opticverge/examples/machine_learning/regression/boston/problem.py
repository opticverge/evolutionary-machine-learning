from typing import Any, Dict

from sklearn import datasets

from opticverge.core.chromosome.abstract_chromosome import AbstractChromosome
from opticverge.core.enum.objective import Objective
from opticverge.core.log.logger import data_logger, DATA
from opticverge.core.solver.abstract_solver import AbstractSolver
from opticverge.external.scikit.enum.normaliser import Normaliser
from opticverge.external.scikit.enum.scoring_function import Scoring
from opticverge.external.scikit.problem.abstract_regression_problem import AbstractRegressionProblem


class BostonHousingProblem(AbstractRegressionProblem):

    def __init__(self, scoring_function: Scoring, normaliser: Normaliser=None, folds: int = 1):
        dataset = datasets.load_boston()

        super(BostonHousingProblem, self).__init__(
            Objective.Minimisation,
            "Boston House Pricing Prediction",
            data_x=dataset.get("data"),
            target_x=dataset.get("target"),
            normaliser=normaliser,
            folds=folds,
            scoring_function=scoring_function
        )

    def log_chromosome(self, chromosome: AbstractChromosome, solver: AbstractSolver, additional_data: Dict[str, Any] = None, separator="|"):

        data_str = super(BostonHousingProblem, self).log_chromosome(
            chromosome,
            solver,
            None
        )

        data_logger.log(DATA, data_str)

    def objective_function(self, chromosome: AbstractChromosome):
        super(BostonHousingProblem, self).objective_function(chromosome)

