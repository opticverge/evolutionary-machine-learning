import unittest

from opticverge.core.generator.int_distribution_generator import rand_poisson


class TestHelpers(unittest.TestCase):

    def test_poisson_type_int(self):

        # GIVEN
        value = 10
        min_val = 2
        max_val = 20
        rounding = None
        sample_size = 10
        output_dtype = int


        # WHEN
        actual = rand_poisson(value, min_val, max_val, rounding, sample_size, output_dtype)

        # THEN
        expected = int
        self.assertIsInstance(actual, expected)

    def test_poisson_params(self):

        # GIVEN
        value = 10
        min_val = 2
        max_val = 20
        rounding = None
        sample_size = 10
        output_dtype = int

        # WHEN
        result = rand_poisson(value, min_val, max_val, rounding, sample_size, output_dtype)

        # THEN
        expected = True
        actual = True
        message = ""

        if result < min_val:
            message = "The result should be greater than the minimum value."
        elif result > max_val:
            message = "The result should be less than the maximum value."
        elif type(result) is not output_dtype:
            message = "The result should be of type {}".format(type(output_dtype))

        self.assertTrue(expected == actual, message)


def run_test():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHelpers)
    unittest.TextTestRunner().run(suite)


if __name__ == "__main__":
    run_test()
