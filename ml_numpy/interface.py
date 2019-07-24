import abc

import numpy as np


class SupervisedModel(abc.ABC):
    @abc.abstractmethod
    def fit(self, data: np.ndarray, target: np.ndarray):
        """

        :param data:
        :param target:
        :return:
        """
        pass

    @abc.abstractmethod
    def predict(self, data: np.ndarray):
        """

        :param data:
        :return:
        """
        pass
