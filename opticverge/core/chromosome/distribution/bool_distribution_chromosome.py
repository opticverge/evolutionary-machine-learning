from collections import OrderedDict
from typing import Dict

from opticverge.core.chromosome.abstract_chromosome import AbstractChromosome
from opticverge.core.chromosome.function_chromosome import FunctionChromosome
from opticverge.core.generator.bool_generator import rand_uniform_bool


class RandUniformBooleanChromosome(FunctionChromosome):
    def blueprint_factory(self, loc) -> Dict[str, AbstractChromosome]:
        return OrderedDict({
            "loc": loc
        })

    def __init__(self, loc=0.5):
        super(RandUniformBooleanChromosome, self).__init__(
            rand_uniform_bool,
            self.blueprint_factory(loc=loc)
        )

