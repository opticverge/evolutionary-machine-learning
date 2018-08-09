
from opticverge.core.enum.policy import Policy
from opticverge.core.log.logger import application_logger
from opticverge.core.solver.generic_ais import AIS
from opticverge.examples.optimisation.rastrigin.chromosome import RastriginChromosome
from opticverge.examples.optimisation.rastrigin.problem import RastriginProblem


def run():
    problem = RastriginProblem()

    try:

        solver = AIS(
            chromosome=RastriginChromosome(dimensions=10),
            problem=problem,
            population_size=100,
            epochs=100,
            policies=[
                Policy.EnforceLimitedMutationAttempts,
                Policy.EnforceUniqueChromosome
            ],
            duration=None
        )

        best_chromosome = solver.run()

        problem.log_chromosome(best_chromosome, solver)

    except Exception as ex:
        application_logger.exception(
            exc_info=ex,
            msg="Exception occurred whilst attempting to solve {}"
                .format(problem.name))


if __name__ == '__main__':
    run()
