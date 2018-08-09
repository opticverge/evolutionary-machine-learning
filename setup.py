from distutils.core import setup

setup(
    name='opticverge-evolutionary-machine-learning',
    version='0.0.1',
    packages=['opticverge', 'opticverge.lib', 'opticverge.core', 'opticverge.core.log', 'opticverge.core.enum',
              'opticverge.core.meta', 'opticverge.core.util', 'opticverge.core.solver', 'opticverge.core.numeric',
              'opticverge.core.problem', 'opticverge.core.strategy', 'opticverge.core.generator',
              'opticverge.core.chromosome', 'opticverge.core.chromosome.distribution', 'opticverge.test',
              'opticverge.examples', 'opticverge.examples.optimisation', 'opticverge.examples.optimisation.ackley',
              'opticverge.examples.optimisation.one_max', 'opticverge.examples.machine_learning',
              'opticverge.examples.machine_learning.regression',
              'opticverge.examples.machine_learning.regression.boston',
              'opticverge.examples.machine_learning.regression.diabetes', 'opticverge.external',
              'opticverge.external.scikit', 'opticverge.external.scikit.enum', 'opticverge.external.scikit.problem',
              'opticverge.external.scikit.chromosome', 'opticverge.external.scikit.chromosome.regression'],
    url='https://github.com/opticverge/evolutionary-machine-learning',
    license='MIT',
    author='Jonathan Hussey',
    author_email='jhusseydesign@gmail.com',
    description='An evolutionary computation library for optimisation and machine learning'
)
