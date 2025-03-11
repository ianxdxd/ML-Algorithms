import numpy as np


class EvaluationMetrics:
    def __init__(self, n: int, y, y_pred):
        self.n = n
        self.y = np.reshape(y, (-1, 1))  # actual value
        self.y_pred = y_pred  # predicted value

    def mae(self):
        return np.mean(np.abs(self.y - self.y_pred))

    def mse(self):
        return sum(np.pow((self.y - self.y_pred), 2)) / self.n

    def rmse(self):
        return np.sqrt(sum(np.pow((self.y - self.y_pred), 2)) / self.n)
