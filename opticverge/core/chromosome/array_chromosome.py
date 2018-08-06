import copy
from collections import OrderedDict
from typing import List

import numpy as np

from opticverge.core.chromosome.abstract_chromosome import AbstractChromosome
from opticverge.core.chromosome.function_chromosome import FunctionChromosome
from opticverge.core.generator.int_distribution_generator import rand_int
from opticverge.core.generator.options_generator import rand_options
from opticverge.core.generator.real_generator import rand_real


class RandArrayChromosome(FunctionChromosome):
    """ The chromosome for generating fixed or dynamic arrays """

    def __init__(self, generator: AbstractChromosome, length: int, fixed: bool = False):
        """ The constructor for this class

        Args:
            generator (AbstractChromosome): An instance of a class derived from an AbstractChromosome
            length (int): The length of the array
            fixed (bool): Whether the array length is fixed
        """
        super(RandArrayChromosome, self).__init__(generator, OrderedDict())

        """
        The generator is responsible for producing each entry within the array
        """
        self.__generator = generator

        """
        The length represents the initial size of the array if fixed is True
        """
        self.__length = length

        """
        Fixed represents whether the array is subject to change during the
        evolutionary process.
        """
        self.__fixed = fixed

    def generate(self, **kwargs):

        # determine the length of the array to generate
        length: int = self.__length if self.__fixed is True else rand_int(1, self.__length)

        # generate each of the positions of the array using the generator
        for i in range(length):
            self.genotype[i] = self.__generator.generate(**kwargs)

        # the phenotype of an array represents the values from the genotype,
        # since we use an OrderedDict as our base representation we are safe to
        # use list(self.genotype.values())
        self.phenotype = list(self.genotype.values())

        return self.phenotype

    def mutate(self, mutation_probability: float, **kwargs):
        """ The mutation function for the array_chromosome

        When mutating the ArrayChromosome we use a number of techniques to
        modify the contents.

        1. Iterate through each entry in the array and with some probability
           change the value based on the generator
        2. Then select pairs of positions at random with some probability and swap
           their values
        3. If fixed is set to False then:
           a) Attempt to add an item to the array with some probability
           b) Attempt to remove an item from the array with some probability

        Args:
            mutation_probability: The likelihood that we will change the array
            **kwargs:

        Returns:

        """

        # 1. Attempt to mutate each value
        for key, val in self.genotype.items():
            if rand_real() < mutation_probability:
                self.genotype[key] = self.__generator.generate(**kwargs)

        # 2. Attempt to swap positions of the array
        keys: List[str or int] = list(self.genotype.keys())
        if len(keys) > 1:

            # select the number of items to modify in the list
            items_to_select: int = rand_int(2, len(keys))

            selected: List[str or int] = rand_options(keys, items_to_select)

            shuffled: List[np.int64] = copy.copy(selected)

            np.random.shuffle(shuffled)

            for i, key in enumerate(selected):

                if rand_real() < mutation_probability:
                    self.genotype[selected[i]], self.genotype[shuffled[i]] = self.genotype[shuffled[i]], self.genotype[
                        selected[i]]

        # TODO: Sensibly define how to insert/update an OrderedDict
        # 3. Attempt to add and remove items to the array
        # if self.__fixed is False:
        #
        #     # Add
        #     num_positions = rand_int(1, len(keys))
        #
        #     # create a temporary placeholder to update the OrderedDict
        #     temp_phenotype = list(self.genotype.values())
        #
        #     for i in range(num_positions):
        #
        #         if rand_real() < mutation_probability:
        #             position = rand_int(0, len(self.genotype.keys()))
        #             temp_phenotype.insert(position, self.__generator.generate(**kwargs))
        #
        #     # Remove

        self.phenotype = list(self.genotype.values())

        return self.phenotype
