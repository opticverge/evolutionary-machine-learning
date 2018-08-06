from collections import OrderedDict
from typing import List, Any

from opticverge.core.chromosome.function_chromosome import FunctionChromosome
from opticverge.core.generator.options_generator import rand_options


class RandOptionsChromosome(FunctionChromosome):
    def __init__(self, options: List[Any], size: int = None, replacement: bool = False):
        super(RandOptionsChromosome, self).__init__(
            rand_options,
            OrderedDict({
                "options": options,
                "size": size,
                "replacement": replacement
            })
        )
