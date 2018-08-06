from enum import Enum
from typing import Callable

from sklearn import metrics


class Scoring(Enum):

    MeanSquaredError = "MeanSquaredError"
    MeanAbsoluteError = "MeanAbsoluteError"

    @staticmethod
    def scoring_function_map():
        return {
            Scoring.MeanSquaredError: metrics.mean_squared_error,
            Scoring.MeanAbsoluteError: metrics.mean_absolute_error
        }

    @staticmethod
    def get_scoring_function(scoring_function: str) -> Callable:
        if scoring_function not in Scoring.scoring_function_map():
            raise NotImplementedError(
                "The evaluation function is not a part of the function map, please link {} to an evaluation function"
                .format(scoring_function)
            )

        return Scoring.scoring_function_map().get(scoring_function)

