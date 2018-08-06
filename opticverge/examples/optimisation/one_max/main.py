from opticverge.core.enum.policy import Policy
from opticverge.core.log.logger import application_logger
from opticverge.core.solver.generic_ais import AIS
from opticverge.examples.optimisation.one_max.chromosome import OneMaxChromosome
from opticverge.examples.optimisation.one_max.problem import OneMaxProblem


def run():
    """
    The OneMax problem represents a classical optimisation problem where the
    chromosome is represented as a list of 1s an 0s and the quality of the 
    chromosome is evaluated by the sum of it's alleles.
    """
    problem = OneMaxProblem()

    try:

        """
        The Artificial Immune System is a bio inspired evolutionary method and
        is used here to demonstrate the application of the framework to the 
        OneMax problem. 
        """
        solver = AIS(
            chromosome=OneMaxChromosome(dimensions=100),
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
            msg="Exception occurred whilst attempting to solve the {}"
                .format(problem.name))


if __name__ == '__main__':
    run()
