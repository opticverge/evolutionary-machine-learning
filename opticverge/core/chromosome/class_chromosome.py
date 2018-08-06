from typing import Callable, Dict, Any

from opticverge.core.chromosome.abstract_chromosome import AbstractChromosome


class ClassChromosome(AbstractChromosome):
    def __init__(self,
                 constructor: Callable,
                 blueprint: Dict[str, AbstractChromosome],
                 fixed_genotype: Dict[str, Any] = None):
        """The constructor for the ClassChromosome
        
        Args:
            constructor (Callable): The reference to the class
            blueprint (Dict[str, AbstractChromosome]): The description on how to generate the genotype
            fixed_genotype (Dict[str, Any], optional): Defaults to None. The values of the genotype that should remain unchanged
        """

        super(ClassChromosome, self).__init__(blueprint=blueprint, fixed_genotype=fixed_genotype)
        self.__constructor = constructor

    def generate(self, **kwargs) -> Any:
        super(ClassChromosome, self).generate()
        self.phenotype = self.__constructor(**{**self.genotype, **kwargs})
        return self.phenotype

    def mutate(self, mutation_probability: float, **kwargs) -> Any:
        super(ClassChromosome, self).mutate(mutation_probability=mutation_probability, **kwargs)
        self.phenotype = self.__constructor(**{**self.genotype, **kwargs})
        return self.phenotype

