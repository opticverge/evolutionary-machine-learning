import concurrent
from concurrent.futures import Future
from math import exp
from typing import List, Dict

import psutil

from opticverge.core.chromosome.abstract_chromosome import AbstractChromosome
from opticverge.core.enum.policy import Policy
from opticverge.core.numeric.convert import scale
from opticverge.core.solver.abstract_solver import AbstractSolver
from opticverge.core.strategy.selection import elitist_selection


class AIS(AbstractSolver):
    """ The Artificial Immune system is an evolutionary search method

    """
    def __init__(self, chromosome, problem, population_size, epochs, policies, duration=None):
        """

        Args:
            chromosome: The chromosome this solver is optimising
            problem: The problem to be solved
            population_size: The size of the population
            epochs: The number of generations to run for
            policies: The policies to abide by during the evolutionary process
            duration: The length of time in seconds to evolve the chromosomes
        """

        super(AIS, self).__init__(
            chromosome=chromosome,
            problem=problem,
            population_size=population_size,
            epochs=epochs,
            policies=policies,
            duration=duration
        )

    def run(self) -> AbstractChromosome:
        return super(AIS, self).run()

    def mutation_probability(self, chromosome: AbstractChromosome) -> float:
        """ Extracts the mutation probability for the current chromosome based on its position within the population

        Args:
            chromosome:

        Returns:
            float
        """

        self.sort_population()

        best_chromosome: AbstractChromosome = self.population[0]
        worst_chromosome: AbstractChromosome = self.population[-1]

        if worst_chromosome is None:
            for population_chromosome in reversed(self.population):
                if population_chromosome.fitness is None:
                    continue
                worst_chromosome = population_chromosome
                break

        return _mutation_probability(chromosome.fitness, best_chromosome.fitness, worst_chromosome.fitness)

    def mutate_population(self):

        self.sort_population()

        with concurrent.futures.ThreadPoolExecutor(max_workers=psutil.cpu_count()) as executor:
            futures: List[Future] = []
            for i, chromosome in enumerate(self.population):
                future: Future = executor.submit(
                    _mutate_chromosome,
                    chromosome=chromosome,
                    mutation_probability=self.mutation_probability(chromosome),
                    amount_to_generate=int(max(round(self.population_size / (i + 1)), 1)),
                    policies=self.policies,
                    existing_chromosomes=self.meta.chromosome_tracker
                )

                futures.insert(i, future)

            for j, future in enumerate(concurrent.futures.as_completed(futures)):
                mutated_chromosomes: List[AbstractChromosome] = future.result()
                self.evaluate_chromosomes(mutated_chromosomes)
                self.sort_chromosomes(mutated_chromosomes)
                self.population[j] = elitist_selection(self.population[j], mutated_chromosomes, self.problem.objective)


def _mutate_chromosome(
        chromosome: AbstractChromosome,
        mutation_probability: float,
        amount_to_generate: int,
        policies: List[Policy] = None,
        attempts: int = 10000,
        existing_chromosomes: Dict[str, AbstractChromosome] = None) -> List[AbstractChromosome]:
    """ This function is not a part of the Solver class as we do not want to serialize the entire class

    Args:
        chromosome:
        mutation_probability:
        amount_to_generate:
        policies:
        attempts:
        existing_chromosomes:

    Returns:

    """
    mutated_chromosomes: Dict[str, AbstractChromosome] = {}
    while len(mutated_chromosomes) < amount_to_generate:

        if Policy.EnforceLimitedMutationAttempts in policies:
            if attempts < 1:
                break

            attempts -= 1

        clone: AbstractChromosome = chromosome.clone()

        clone.mutate(mutation_probability)

        if Policy.EnforceUniqueChromosome in policies:

            if chromosome.id == clone.id is True:
                continue

            if clone.id in mutated_chromosomes:
                continue

            if clone.id in existing_chromosomes:
                continue

            mutated_chromosomes[clone.id] = clone
        else:
            mutated_chromosomes[len(mutated_chromosomes)] = clone

    return list(mutated_chromosomes.values())


def _mutation_probability(chromosome_fitness: int or float, best_chromosome_fitness: int or float,
                          worst_chromosome_fitness: int or float):
    scaled_fitness: float = 0.5

    if best_chromosome_fitness is not None and worst_chromosome_fitness is not None and chromosome_fitness is not None:
        scaled_fitness = scale(chromosome_fitness, worst_chromosome_fitness, best_chromosome_fitness, 0., 1.)

    return exp(-2.5 * scaled_fitness)
