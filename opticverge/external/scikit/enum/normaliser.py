from enum import Enum
from typing import Callable

from sklearn.preprocessing import StandardScaler, RobustScaler


class Normaliser(Enum):

    StandardScaler = 'StandardScaler'
    RobustScaler = 'RobustScaler'

    @staticmethod
    def normaliser_map():
        return {
            Normaliser.StandardScaler: StandardScaler,
            Normaliser.RobustScaler: RobustScaler
        }

    @staticmethod
    def get_normaliser(normaliser: str, **kwargs) -> Callable:
        """ Retrieves an instance of the normaliser from the scikit learn preprocessing package

        Args:
            normaliser(Normaliser): The enum variant of the normaliser instance to retrieve
            **kwargs: key word arguments passed to the constructor

        Returns:

        """
        if normaliser not in Normaliser.normaliser_map():
            raise NotImplementedError(
                "Please add the mapping between the normaliser and the constructor for {}"
                .format(normaliser)
            )
        normaliser_constructor = Normaliser.normaliser_map().get(normaliser)
        return normaliser_constructor(**kwargs)
