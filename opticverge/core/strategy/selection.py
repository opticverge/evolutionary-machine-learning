from typing import List

from opticverge.core.chromosome.abstract_chromosome import AbstractChromosome
from opticverge.core.enum.objective import Objective


def elitist_selection(chromosome: AbstractChromosome, mutated_chromosomes: List[AbstractChromosome],
                      objective: Objective) -> AbstractChromosome:
    """ Selects a chromosome using the elitist strategy

    Args:
        chromosome (AbstractChromosome): The source chromosome
        mutated_chromosomes (List[AbstractChromosome]): The mutated chromosomes originating from the chromosome
        objective (Objective): The objective of the problem

    Returns:
        AbstractChromosome: The selected chromosome based on the elitist strategy

    """
    if mutated_chromosomes is None or len(mutated_chromosomes) == 0:
        return chromosome

    best_chromosome: AbstractChromosome = mutated_chromosomes[0]

    if best_chromosome.fitness is None and chromosome.fitness is not None:
        return chromosome

    if best_chromosome.fitness is not None and chromosome.fitness is None:
        return best_chromosome

    if best_chromosome.fitness is None and chromosome.fitness is None:
        return chromosome

    if objective is Objective.Maximisation:
        if best_chromosome.fitness >= chromosome.fitness:
            return best_chromosome
    else:
        if best_chromosome.fitness <= chromosome.fitness:
            return best_chromosome

    return chromosome
