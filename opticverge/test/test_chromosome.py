import unittest
from collections import OrderedDict
from typing import List

import numpy as np

from opticverge.core.chromosome.array_chromosome import RandArrayChromosome
from opticverge.core.chromosome.distribution.int_distribution_chromosome import RandIntChromosome
from opticverge.core.generator.int_distribution_generator import rand_poisson


class TestHelpers(unittest.TestCase):

    def test_array_chromosome(self):

        # GIVEN
        chromosome = RandArrayChromosome(
            RandIntChromosome(0, 10),
            length=10,
            fixed=True
        )

        # WHEN
        result: List[np.int64] = chromosome.generate()

        # THEN
        expected = 10
        actual = len(result)
        self.assertTrue(actual == expected)


def run_test():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHelpers)
    unittest.TextTestRunner().run(suite)


if __name__ == "__main__":
    run_test()
