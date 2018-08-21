from typing import Any, Dict

import numpy as np
import pandas

from opticverge.core.chromosome.abstract_chromosome import AbstractChromosome
from opticverge.core.enum.objective import Objective
from opticverge.core.log.logger import data_logger, DATA
from opticverge.core.solver.abstract_solver import AbstractSolver
from opticverge.external.scikit.enum.normaliser import Normaliser
from opticverge.external.scikit.enum.scoring_function import Scoring
from opticverge.external.scikit.problem.abstract_regression_problem import AbstractRegressionProblem


class ForestFiresPredictionProblem(AbstractRegressionProblem):
    def __init__(self, scoring_function: Scoring, normaliser: Normaliser = None, folds: int = 1):
        data = []
        target = []

        df = pandas.read_csv("./forestfires.csv", sep=",",
                             usecols=["X", "Y", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain", "area"])

        for i, row in df.iterrows():
            data.append(row.values[:-1])
            target.append(row.values[-1])

        #
        data = np.array(data)
        target = np.array(target)

        super(ForestFiresPredictionProblem, self).__init__(
            Objective.Minimisation,
            "Forest Fires Prediction",
            data_x=data,
            target_x=target,
            normaliser=normaliser,
            folds=folds,
            scoring_function=scoring_function
        )

    def log_chromosome(self, chromosome: AbstractChromosome, solver: AbstractSolver,
                       additional_data: Dict[str, Any] = None, separator="|"):
        data_str = super(ForestFiresPredictionProblem, self).log_chromosome(
            chromosome,
            solver,
            None
        )

        data_logger.log(DATA, data_str)

    def objective_function(self, chromosome: AbstractChromosome):
        super(ForestFiresPredictionProblem, self).objective_function(chromosome)
