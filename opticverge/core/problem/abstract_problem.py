import re
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, TypeVar

from opticverge.core.chromosome.abstract_chromosome import AbstractChromosome
from opticverge.core.enum.objective import Objective
from opticverge.core.solver.abstract_solver import AbstractSolver

AbstractSolverEntity = TypeVar('AbstractSolver')


class AbstractProblem(metaclass=ABCMeta):
    def __init__(self, objective: Objective, name: str):
        """The default constructor for this class
        
        Args:
            objective (Objective): Set whether this problem is maximising or minimising the objective function
            name (str): The name of the problem to be solved
        """

        self.__objective = objective
        self.__problem_name = name

    @property
    def name(self) -> str:
        """Get the name of the problem to be solved
        
        Returns:
            str: The problem name
        """
        return self.__problem_name

    @property
    def objective(self) -> Objective:
        """Get the objective of the problem
        
        Returns:
            Objective: The problem objective
        """
        return self.__objective

    @abstractmethod
    def objective_function(self, chromosome: AbstractChromosome):
        """Evaluates the quality of a chromosome
        
        Args:
            chromosome (AbstractChromosome): The chromosome to measure
        
        Raises:
            NotImplementedError: Must be implemented by the derived class
        """
        chromosome.meta.evaluated = True

    @abstractmethod
    def log_chromosome(self, chromosome: AbstractChromosome, solver: AbstractSolverEntity,
                       additional_data: Dict[str, Any] = None, separator="|") -> str:
        """Produces a string formatted for logging to a log file
        
        Args:
            chromosome (AbstractChromosome): The generated chromosome
            solver (AbstractSolver): The solver navigating the search space
            additional_data (Dict[str, Any], optional): Defaults to None. Additional properties to be logged
            separator (str, optional): Defaults to "|". The data separator when logged to a file
        
        Returns:
            str: The data formatted to a string e.g. date|id|fitness ...
        """

        log_data = OrderedDict({
            "problem_name": self.name,
            "solver_id": solver.meta.id,
            "chromosome": solver.chromosome.__class__.__name__,
            "fitness": chromosome.fitness,
            "generation": solver.generation,
            "chromosome_id": chromosome.meta.id,
            "parent_id": chromosome.meta.parent_id,
            "phenotype": re.sub("[\r\n\\s]+", " ", "{}".format(dict(chromosome.genotype)))
        })

        if additional_data is not None:
            log_data.update(additional_data)

        placeholder_string = ["{}" for i in range(len(log_data))]

        return separator.join(placeholder_string).format(*log_data.values())
