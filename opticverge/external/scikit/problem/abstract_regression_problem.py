import concurrent.futures
from abc import ABCMeta
from collections import OrderedDict
from typing import List, Callable, Dict, Any

import numpy as np
import psutil
from sklearn.model_selection import train_test_split, KFold

from opticverge.core.chromosome.abstract_chromosome import AbstractChromosome
from opticverge.core.globals import DEFAULT_NUM_JOBS
from opticverge.core.log.logger import application_logger
from opticverge.core.problem.abstract_problem import AbstractProblem
from opticverge.core.enum.objective import Objective
from opticverge.core.solver.abstract_solver import AbstractSolver
from opticverge.external.scikit.enum.normaliser import Normaliser
from opticverge.external.scikit.enum.scoring_function import Scoring


class AbstractRegressionProblem(AbstractProblem, metaclass=ABCMeta):

    def __init__(self,
                 objective: Objective,
                 name: str,
                 data_x: np.array or List,
                 target_x: np.array or List,
                 scoring_function: Scoring,
                 folds: int = 1,
                 train_test_ratio: float = 1.0,
                 normaliser: Normaliser or Callable = None):

        super(AbstractRegressionProblem, self).__init__(objective, name)

        self.__data_x = data_x
        self.__target_x = target_x
        self.__scoring_function_enum = scoring_function
        self.__scoring_function = None if scoring_function is None else Scoring.get_scoring_function(scoring_function)
        self.__folds = folds
        self.__normaliser_enum = normaliser
        self.__train_test_ratio = train_test_ratio

        self.__partitioned_data = None
        self.__normalised_data = None

    @property
    def data(self, **kwargs):

        # if the data has not already been normalised then apply normalisation
        # strategy
        if self.__normalised_data is None:

            # apply the ratio of training to the number of items in the data
            training_data_size = int(self.__train_test_ratio * len(self.__data_x))

            # initially assign the non-normalised data
            self.__normalised_data = self.__data_x[:training_data_size]

            # if the user has decided to use one of the normalisation strategies
            # then apply it
            if self.__normaliser_enum is not None:

                # lookup callable normaliser from scikit learn library
                normaliser: Callable = Normaliser.get_normaliser(self.__normaliser_enum, **kwargs)

                # transform the data and assign it to the normalised data
                self.__normalised_data = normaliser.fit_transform(self.__data_x, self.__target_x)

        return self.__normalised_data

    @property
    def partitions(self, dtype=np.float64, test_size=0.1):

        # if we haven't already partitioned the data then do so
        if self.__partitioned_data is None:

            self.__partitioned_data = []

            # get the normalised data and its target variables
            data_x = self.data
            target_x = self.__target_x

            # if the user prefers not to use a validation strategy then split
            # the data into a single fold.
            if self.__folds < 2 or self.__folds is None:
                x_train, x_test, y_train, y_test = train_test_split(data_x, target_x, test_size=test_size)
                self.__partitioned_data.append(data_object(x_train, y_train, x_test, y_test, dtype))
            else:
                # use the default KFold validation strategy
                # TODO: expose alternative folding strategies
                validation_strategy = KFold(n_splits=self.__folds, shuffle=True)

                # split the data and build the partitions to match the number
                # of folds
                cv_split = validation_strategy.split(data_x, target_x)
                for train_index, test_index in cv_split:
                    x_train, x_test = data_x[train_index], data_x[test_index]
                    y_train, y_test = target_x[train_index], target_x[test_index]
                    self.__partitioned_data.append(data_object(x_train, y_train, x_test, y_test, dtype))

        return self.__partitioned_data

    def objective_function(self, chromosome: AbstractChromosome) -> List[np.float64] or None:

        # by default the scores are set to None given that an exception could
        # occur during the learning phase
        scores = None

        try:

            # we prefer to farm out the learning to the number of cores assigned
            # in DEFAULT_NUM_JOBS
            with concurrent.futures.ProcessPoolExecutor(max_workers=DEFAULT_NUM_JOBS) as executor:

                # here we track the result of the training against the learners
                futures = []

                for i, partition in enumerate(self.partitions):

                    # The phenotype of the chromosome represents an instance of
                    # a learner that implements the fit function. The learn
                    # function is not a part of this class simply because we
                    # do not want to serialize the entire class in order to
                    # call the method
                    future = executor.submit(
                        learn,
                        learner=chromosome.phenotype,
                        partition=partition,
                        evaluation_function=self.__scoring_function
                    )

                    # add the futures
                    futures.insert(i, future)

                # wait until the futures are complete
                concurrent.futures.wait(futures)

                # extract the output of each future into a list
                scores = [future.result() for future in futures]

        except Exception as ex:
            application_logger.exception(
                "AbstractRegressionProblem-objective_function: Failure during learning phase",
                exc_info=ex
            )

        # by default uses the average of the the scores from k-fold validation
        chromosome.fitness = None if scores is None else np.mean(scores)

        # call the base class objective function
        super(AbstractRegressionProblem, self).objective_function(chromosome)

        # provides the result of cross validation allowing the caller to decide
        # what they want to do after the evaluation
        return scores

    def log_chromosome(self, chromosome: AbstractChromosome, solver: AbstractSolver, additional_data: Dict[str, Any] = None, separator="|"):

        additional_data = OrderedDict({
            "normaliser": self.__normaliser_enum,
            "evaluation_function": self.__scoring_function_enum,
            "folds": self.__folds
        })

        return super(AbstractRegressionProblem, self).log_chromosome(chromosome, solver, additional_data, separator)


def learn(learner, partition, evaluation_function, **kwargs):
    learner.fit(X=partition.get("x_train"), y=partition.get("y_train"), **kwargs)
    predictions = list(learner.predict(partition.get("x_test")))
    return evaluation_function(partition.get("y_test"), predictions)


def data_object(x_train, y_train, x_test, y_test, dtype=np.float64):
    return {
        "x_train": np.array(x_train, dtype=dtype),
        "y_train": np.array(y_train, dtype=dtype),
        "x_test": np.array(x_test, dtype=dtype),
        "y_test": np.array(y_test, dtype=dtype)
    }
