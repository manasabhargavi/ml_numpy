import numpy as np

from ml_numpy import interface


class LinearRegressor(interface.SupervisedModel):
    def fit(self, data: np.ndarray, target: np.ndarray):
        pass

    def predict(self, data: np.ndarray):
        pass
