import numpy as np
from Utils.weights import Weights


class Regression:
    def __init__(self, n_iter: int = 100, l_rate: float = 0.001):
        self.n_iter = n_iter
        self.l_rate = l_rate
        self.w = None
        self.b = 0
        self.error = 0

    def regularize(self, lamb: float = 0):
        return {
            None: 0,
            "L1": lamb * np.sign(self.w),
            "L2": 2 * lamb * self.w,
        }

    def predict(self, X):
        return np.dot(X, self.w) + self.b

    def fit(self, X, y, reg_factor: str = None):
        """
        i -> number of samples
        j -> number of features
        """
        n_samples, n_features = X.shape
        print(self.w)
        if self.w is None:  # persisting weights
            self.w = Weights(n_features).random_uniform()
        print(self.w)
        y = np.reshape(y, (-1, 1))  # transform an array to a column vector
        regularization = self.regularize()

        for _ in range(self.n_iter):
            y_pred = self.predict(X)
            self.error = y - y_pred
            grad_w = (-2 * np.dot(X.T, self.error) / n_samples) + regularization.get(
                reg_factor, 0
            )
            grad_b = (-2 * np.sum(self.error)) / n_samples
            self.w -= self.l_rate * grad_w  # l_rate * (grad_w + regularization_factor)
            self.b -= self.l_rate * grad_b  # l_rate * grad_b


class LinearRegression(Regression):
    def __init__(self, n_iter: int, l_rate: float):
        super().__init__(n_iter, l_rate)

    def printWeights(self):
        print(self.w, self.w.shape)

    def gradientDescent(self, X, y):
        super().fit(X, y)

    def leastSquares(X, y):
        """
        This algorithm works by making the total of the squares of the erros as
        small as possible. However, this model is sensitive to outliers, adding bias
        and pulling the line towards it.
        """
        N, _ = X.shape
        y = y.reshape(-1, 1)

        m = N * (sum(X * y) - sum(X) * sum(y)) / N * np.pow(sum(X), 2) - np.pow(
            sum(X), 2
        )
        b = (sum(y) - m * sum(X)) / N
        return m, b

    def ordinaryLeastSquares(X, y):
        pass

    def moorePenroseLeastSquares(X, y):
        pass
