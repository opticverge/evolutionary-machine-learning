import copy
from typing import TypeVar

"""Typing for the chromosome meta class to be returned"""
ChromosomeMetaEntity = TypeVar('ChromosomeMeta')


class ChromosomeMeta(object):

    def __init__(self):
        """The constructor for this class
        """

        # a unique identifer for this chromosome, typically derived from the
        # genotype of the chromosome
        self.__id: str = None

        # the time taken in milliseconds to evaluate the chromosome
        self.__evaluation_time: int = None

        # the parent chromosome id
        self.__parent_id: str = None

        # tracks whether the chromosome was evaluated
        self.__evaluated: bool = False
    
    @property
    def evaluated(self) -> bool:
        """Get whether the chromosome was evaluated
        
        Returns:
            bool: The value
        """
        return self.__evaluated

    @evaluated.setter
    def evaluated(self, value: bool):
        """Set the value of evaluated
        
        Args:
            value (bool): The evaluated value
        """
        self.__evaluated = value
    
    @property
    def parent_id(self) -> str:
        """Get chromosome parent id
        
        Returns:
            str: The parent id
        """
        return self.__parent_id

    @parent_id.setter
    def parent_id(self, parent_id: str):
        """Set the value of parent_id
        
        Args:
            parent_id (str): The parent_id value
        """
        self.__parent_id = parent_id
    
    @property
    def id(self) -> str:
        """Get the id of the chromosome derived from the phenotype of the chromosome
        
        Returns:
            str: The chromosome id
        """
        return self.__id

    @id.setter
    def id(self, value: str):
        """Set the id of the chromosome
        
        Args:
            value (str): The id 
        """
        self.__id = value
    
    @property
    def evaluation_time(self) -> int:
        """Get the evaluation_time of the chromosome
        
        Returns:
            int: The chromosome evaluation time
        """
        return self.__evaluation_time

    @evaluation_time.setter
    def evaluation_time(self, value: int):
        """Set the evaluation_time of the chromosome
        
        Args:
            value (int): The evaluation time 
        """
        self.__evaluation_time = value

    def clone(self) -> ChromosomeMetaEntity:

        clone = copy.deepcopy(self)
        clone.__id = None
        clone.__parent_id = self.id
        clone.__evaluation_time = None
        clone.__evaluated = False

        return clone 
