import copy
from abc import ABCMeta
from collections import OrderedDict
from typing import Any, Dict, TypeVar

import xxhash

from opticverge.core.generator.real_generator import rand_real
from opticverge.core.meta.chromosome_meta import ChromosomeMeta

"""
Typing for the chromosome meta class since we can't use the AbstractChromosome
class reference within itself...
"""
AbstractChromosomeEntity = TypeVar('AbstractChromosome')


class AbstractChromosome(metaclass=ABCMeta):
    """ The base class for all chromosomes """

    def __init__(self, blueprint: Dict[str, AbstractChromosomeEntity], fixed_genotype: Dict[str, Any] = None):
        """The base constructor for the derived class

        Args:
            blueprint (Dict[str, AbstractChromosome]): The blue print of creation and change for this chromosome
            fixed_genotype (Dict[str, Any], optional): Defaults to None. The fixed parameters of the genotype
        """

        """
        Given that the blueprint could represent anything, using an OrderedDict
        allows us to guarantee cross compatibility when selecting representations        
        """
        if type(blueprint) is not OrderedDict:
            raise TypeError(
                "Expected blueprint to be of type OrderedDict, received type {}".format(type(blueprint))
            )

        """
        The blueprint contains a key value association where the value
        represents an AbstractChromosome that is a type of generator. The
        AbstractChromosome is essentially something which produces an output
        when the generate function is called.
        """
        self.__blueprint: Dict[str, AbstractChromosomeEntity] = blueprint

        """
        The genotype represents the result of generating a result for each of
        the generators in the blueprint.
        """
        self.__genotype: Dict[str, AbstractChromosomeEntity or Any] = OrderedDict()

        """
        The fixed genotype represents the key value pairs that do not undergo 
        any change or that do not require evolutionary methods.
        """
        self.__fixed_genotype: Dict[str, Any] = fixed_genotype if fixed_genotype is not None else {}

        """
        The phenotype represents the output of the activity from the generation
        of the genotype into whatever is generated from the process. This could 
        be anything, a number, a string, an array, an instance of a class, or
        the output of a function.  
        """
        self.__phenotype = None

        """
        The fitness represents the result of evaluating the chromosome based on
        the objective function defined by the problem
        """
        self.__fitness: int or float = None

        """
        The meta of a chromosome refers to additional data that is tracked in
        relation to the chromosome. This could be an id, whether the chromosome
        was evaluated or who the chromosome descended from after the operation
        of a mutation.
        """
        self.__meta = ChromosomeMeta()

    @property
    def blueprint(self) -> Dict[str, AbstractChromosomeEntity]:
        return self.__blueprint

    @property
    def fitness(self) -> int or float:
        """Get the fitness of the chromosome
        
        Returns:
            int or float: The fitness of the chromosome
        """
        return self.__fitness

    @property
    def genotype(self) -> Dict[str, Any]:
        """Get the genotype of the chromosome
        
        Returns:
            Dict[str, Any]: The genotype description
        """
        return self.__genotype

    @genotype.setter
    def genotype(self, genotype: Dict[str, Any]):
        self.__genotype = genotype

    @property
    def phenotype(self) -> Dict[str, Any]:
        """Get the phenotype of the chromosome
        
        Returns:
            Dict[str, Any]: The result of processing the genotype
        """
        return self.__phenotype

    @phenotype.setter
    def phenotype(self, phenotype: Any):
        self.__phenotype = phenotype

    @fitness.setter
    def fitness(self, value: int or float):
        """Set the fitness of the chromosome
        
        Args:
            value (int or float): The fitness of the chromosome
        """
        self.__meta.evaluated = True
        self.__fitness = value

    @property
    def meta(self):
        return self.__meta

    @property
    def id(self):
        if self.__meta.id is None:
            self.__meta.id = xxhash.xxh64("%s" % self.__genotype).hexdigest()
        return self.__meta.id

    def clone(self) -> AbstractChromosomeEntity:
        """Generates a copy of the chromosome
        
        Returns:
            AbstractChromosome: A copy of the chromosome
        """
        clone = copy.deepcopy(self)
        # clone.__phenotype = None
        clone.__meta = clone.__meta.clone()
        return clone

    def generate(self, **kwargs) -> Dict[str, Any]:
        """Generates the genotype from the blueprint recursively
        
        Returns:
            Dict[str, Any]: The updated genotype
        """

        for identifier, generator in self.__blueprint.items():
            self.__genotype[identifier] = generator.generate(**kwargs)

        self.__genotype.update(self.__fixed_genotype)

        return self.__genotype

    def mutate(self, mutation_probability: float, **kwargs):
        """Mutates the genotype
        
        Args:
            mutation_probability (float): The likelihood of change
        """

        for identifier, generator in self.__blueprint.items():
            if rand_real() < mutation_probability:
                self.__genotype[identifier] = generator.generate(**kwargs)

        self.__genotype.update(self.__fixed_genotype)

        return self.__genotype

    def genotype_factory(self, **kwargs) -> Dict[str, AbstractChromosomeEntity]:
        return OrderedDict({})

    def generate_genotype(self):
        self.__blueprint = self.genotype_factory()
