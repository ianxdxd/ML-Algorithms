import numpy as np
from Utils.weights import Weights


class Regression:
    def __init__(self, n_iter: int, l_rate: float):
        self.n_iter = n_iter
        self.l_rate = l_rate
        self.w = Weights()
        self.b = 0
        self.error = 0

    def predict(self, X):
        return np.dot(X, self.w) + self.b

    def fit(self, X, y):
        return 0
