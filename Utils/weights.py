import numpy as np


class Weights:
    """
    Weights are best initialized as arrays since each element corresponds to the weight of a specific Xi.
    If represented as a column vector (n x 1), proper reshaping is required to ensure a correct dot product.
    """

    def __init__(self):
        pass

    def random_normal(self, n_features: int):
        return np.random.normal(0, 1, (n_features,))

    def random_uniform(self, w, n_features: int):
        lim = 1 / np.sqrt(n_features)
        return np.random.uniform(-lim, +lim, (n_features,))
