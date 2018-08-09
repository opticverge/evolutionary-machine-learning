import copy
from abc import ABCMeta, abstractmethod
from math import ceil
from typing import Dict, List, TypeVar

from opticverge.core.chromosome.abstract_chromosome import AbstractChromosome
from opticverge.core.enum.objective import Objective
from opticverge.core.enum.policy import Policy
# from opticverge.core.problem.abstract_problem import AbstractProblem
from opticverge.core.log.logger import application_logger
from opticverge.core.meta.solver_meta import SolverMeta
from opticverge.core.util.exception import reset_signal, signal_ttl, register_signal, TimeoutException

AbstractProblem = TypeVar('AbstractProblem')


class AbstractSolver(metaclass=ABCMeta):
    """The base class for all solvers"""

    def __init__(self,
                 chromosome: AbstractChromosome,
                 problem: AbstractProblem,
                 population_size: int = 100,
                 epochs: int = 100,
                 policies: List[Policy] or None = None,
                 duration: int or None = None):
        """The constructor for the AbstractSolver
        
        Args:
            chromosome (AbstractChromosome): The chromosome that will be evolved to solve the problem
            problem (AbstractProblem): The problem to be solved
            population_size (int, optional): Defaults to 100. The number of chromosomes to evolve in the population
            epochs (int, optional): Defaults to 100. The number of generations to run for before stopping the evolutionary process
            policies (List[Policy] or None, optional): Defaults to None. The list of policies to enforce for the problem
            duration (int or None, optional): Defaults to None. The number of seconds the solver should run for
        """

        self.__chromosome = chromosome
        self.__problem = problem
        self.__epochs = epochs
        self.__population_size = population_size
        self.__generation: int = 0
        self.__population: List[AbstractChromosome] = []
        self.__duration = duration
        self.__policies = policies or []
        self.__meta = SolverMeta()

    """ GETTERS """

    @property
    def population_size(self):
        return self.__population_size

    @property
    def meta(self) -> SolverMeta:
        return self.__meta

    @property
    def generation(self) -> int:
        """Get the current generation of the solver
        
        Returns:
            int: The current generation
        """
        return self.__generation

    @property
    def population(self) -> List[AbstractChromosome]:
        """Get the population of chromosomes
        
        Returns:
            List[AbstractChromosome]: The population of chromosomes
        """
        return self.__population

    @property
    def duration(self) -> int:
        """Get the set duration of the solver in seconds
        
        Returns:
            int: The duration in seconds that the evolutionary process is running for
        """
        return self.__duration

    @property
    def problem(self) -> AbstractProblem:
        return self.__problem

    @property
    def chromosome(self) -> AbstractChromosome:
        return self.__chromosome

    @property
    def policies(self) -> List[Policy]:
        return self.__policies

    """ ABSTRACT METHODS """

    @abstractmethod
    def run(self) -> AbstractChromosome:
        """Performs the evolutionary process by navigating the search space of the chromosome combination space
        
        Raises:
            NotImplementedError: Must be implemented by the derived class
        
        Returns:
            AbstractChromosome: The chromosome with the best fitness 
        """

        try:
            self.initialise()

            if self.duration is not None:

                try:
                    register_signal()
                    signal_ttl(self.duration)

                    while True:
                        self.__generation += 1
                        self.evolve()

                except TimeoutException:
                    reset_signal()
            else:
                while self.__epochs == -1 or self.generation < self.__epochs:
                    self.__generation += 1
                    self.evolve()
        except KeyboardInterrupt:
            application_logger.info(msg="Keyboard interrupt received, exiting simulation")
        except Exception as ex:
            application_logger.exception(exc_info=ex, msg="Exception occurred whilst attempting to solve the {}".format(
                self.problem.name))

        self.sort_population()
        return self.population[0]

    def mutate(self):
        self.mutate_population()

    @abstractmethod
    def mutate_population(self):
        raise NotImplementedError(
            "You must implement the mutate_population method for your solver"
        )

    # @abstractmethod
    # def mutate_chromosome(self, chromosome: AbstractChromosome):
    #     raise NotImplementedError(
    #         "You must implement the mutate_chromosome method for your solver"
    #     )

    """ LIFECYCLE METHODS """

    def initialise(self):
        """Generates, scores and sorts the initial population
        """
        self.__create_population()
        self.__evaluate_population()
        self.sort_population()

    def generate_chromosome(self) -> AbstractChromosome:
        """Creates a copy of the base chromosome and returns a newly generated chromosome
        
        Returns:
            AbstractChromosome: The generated chromosome
        """

        chromosome: AbstractChromosome = copy.deepcopy(self.__chromosome)

        chromosome.generate()

        return chromosome

    def generate_chromosomes(self, count: int = 100) -> List[AbstractChromosome]:
        """Creates a list of chromosomes
            count (int, optional): Defaults to 100. The number of chromosomes to generate
        
        Returns:
            List[AbstractChromosome]: The generated chromosomes
        """

        chromosomes: Dict[str, AbstractChromosome] = {}

        generated_count: int = 0

        while generated_count < count:

            chromosome: AbstractChromosome = self.generate_chromosome()

            id: int = generated_count

            if Policy.EnforceUniqueChromosome in self.policies:

                id = chromosome.id

                if id in chromosomes or id in self.meta.chromosome_tracker:
                    continue

            chromosomes[id] = chromosome

            generated_count += 1

        return list(chromosomes.values())

    def evaluate_chromosome(self, chromosome: AbstractChromosome):
        """Evaluates a chromosome using the objective function of the chromosome
        
        Args:
            chromosome (AbstractChromosome): [description]
        """

        if chromosome.meta.evaluated is False:

            self.__problem.objective_function(chromosome)

            if Policy.EnforceUniqueChromosome in self.policies:
                self.__meta.chromosome_tracker[chromosome.id] = chromosome

            self.__problem.log_chromosome(chromosome, self)

    def evaluate_chromosomes(self, chromosomes: List[AbstractChromosome]):
        """Evaluates the list of chromosomes

        The evaluation is synchronous by default given that the evaluation
        mechanism may utilise multiple threads.
        
        Args:
            chromosomes (List[AbstractChromosome]): [description]
        """
        for i, chromosome in enumerate(chromosomes):
            self.evaluate_chromosome(chromosome)

    def sort_chromosomes(self, chromosomes: List[AbstractChromosome]):
        """Sorts a list of chromosomes according to the objective function of the problem
        
        Args:
            chromosomes (List[AbstractChromosome]): The chromosomes to sort
        """

        sort_direction = True

        if self.__problem.objective is Objective.Minimisation:
            sort_direction = False

        chromosomes.sort(key=lambda c: (c.fitness is None, c.fitness), reverse=sort_direction)

    def __create_population(self):
        """Creates a population of chromosomes based on the population size
        """
        self.__population = self.generate_chromosomes(self.__population_size)

    def __evaluate_population(self):
        """Evaluates the population of chromosomes
        """
        self.evaluate_chromosomes(self.__population)

    def sort_population(self):
        """Sorts the population of chromosomes
        """
        self.sort_chromosomes(self.__population)

    def evolve(self):
        self.mutate()
        self.replace()

    def replace(self, ratio=0.1):
        amount_to_replace = int(ceil(self.__population_size * ratio) + self.__population_size - len(self.__population))
        replacement_count: int = max(1, amount_to_replace)

        self.__population = list(self.population[:-1 * replacement_count])

        new_chromosomes = self.generate_chromosomes(replacement_count)

        self.evaluate_chromosomes(new_chromosomes)

        self.__population.extend(new_chromosomes)
