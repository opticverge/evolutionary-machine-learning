import uuid
from typing import Dict

from opticverge.core.chromosome.abstract_chromosome import AbstractChromosome


class SolverMeta(object):
    """ The SolverMeta class tracks additional data within the evolutionary process"""
    
    def __init__(self):

        """
        The id of a solver is set automatically
        """
        self.__id: str = str(uuid.uuid4())

        self.__chromosome_count = 0

        """
        Tracks the generated chromosomes during the lifecycle of the evolutionary
        process.
        """
        self.__chromosome_tracker: Dict[str, AbstractChromosome] = {}
        
    @property
    def id(self) -> str:
        return self.__id

    @property 
    def chromosome_tracker(self) -> Dict[str, AbstractChromosome]:
        return self.__chromosome_tracker
    
    @chromosome_tracker.setter
    def chromosome_tracker(self, chromosome_id: str, chromosome: AbstractChromosome):
        self.__chromosome_tracker[chromosome_id] = chromosome
