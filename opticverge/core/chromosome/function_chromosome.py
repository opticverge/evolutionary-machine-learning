from typing import Any, Dict, Callable

from opticverge.core.chromosome.abstract_chromosome import AbstractChromosome


class FunctionChromosome(AbstractChromosome):
    def __init__(self, func: Callable or AbstractChromosome, blueprint: Dict[str, AbstractChromosome or Any],
                 fixed_genotype: Dict[str, Any] = None):
        """The constructor for the FunctionChromosome
        
        Args:
            func (Callable): The function to call on generate and mutate
            blueprint (Dict[str, AbstractChromosome or Any]): The descriptor on how to generate the genotype
            fixed_genotype (Dict[str, Any], optional): Defaults to None. The fixed genotype values
        """

        self.__function = func
        super(FunctionChromosome, self).__init__(blueprint=blueprint, fixed_genotype=fixed_genotype)

        self.genotype = self.blueprint

    def generate(self, **kwargs) -> Any:
        self.phenotype = self.__function(**{**dict(self.genotype), **kwargs})
        return self.phenotype

    def mutate(self, mutation_probability: float, **kwargs):
        return super(FunctionChromosome, self).mutate(mutation_probability=mutation_probability, **kwargs)
