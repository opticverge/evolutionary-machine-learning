from opticverge.core.enum.policy import Policy
from opticverge.core.log.logger import application_logger
from opticverge.core.solver.generic_ais import AIS
from opticverge.examples.machine_learning.regression.diabetes.problem import DiabetesPredictionProblem
from opticverge.external.scikit.chromosome.regression.ensemble import XGBRegressorChromosome
from opticverge.external.scikit.chromosome.regression.neighbor import KNeighborsRegressorChromosome
from opticverge.external.scikit.enum.normaliser import Normaliser
from opticverge.external.scikit.enum.scoring_function import Scoring


def run():
    try:
        problem = DiabetesPredictionProblem(
            scoring_function=Scoring.MeanAbsoluteError,
            normaliser=Normaliser.RobustScaler,
            folds=2
        )

        """
        Regressor choices:
        - AdaBoostRegressorChromosome
        - DecisionTreeRegressorChromosome
        - GradientBoostingRegressorChromosome
        - KNeighborsRegressorChromosome
        - MLPRegressorChromosome
        - RandomForestRegressorChromosome
        - XGBRegressorChromosome
        """
        solver = AIS(
            chromosome=KNeighborsRegressorChromosome(),
            problem=problem,
            population_size=10,
            epochs=2,
            policies=[
                Policy.EnforceLimitedMutationAttempts,
                Policy.EnforceUniqueChromosome
            ],
            duration=None
        )

        best_chromosome = solver.run()

        problem.log_chromosome(best_chromosome, solver)

    except Exception as ex:
        application_logger.exception(msg="", exc_info=ex)


if __name__ == '__main__':
    run()
