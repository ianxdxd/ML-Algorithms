import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from Utils.evaluation_metrics import EvaluationMetrics
from Supervised.regression import LinearRegression


def regressions(df_path: str = None, validation_df_path: str = None, target: str = "y"):
    X, y = make_regression(n_samples=1000, n_features=1, noise=15, random_state=4)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    lr = LinearRegression(1000, 0.01)
    lr.gradientDescent(X_train, y_train)
    lr.printWeights()
    y_pred = lr.predict(X_train)

    MAE = EvaluationMetrics(n=X_train.shape[0], y=y_train, y_pred=y_pred).mae()
    print(f"MAE: {MAE}")

    MSE = EvaluationMetrics(n=X_test.shape[0], y=y_train, y_pred=y_pred).mse()
    print(f"MSE: {MSE}")

    RMSE = EvaluationMetrics(n=X_test.shape[0], y=y_train, y_pred=y_pred).rmse()
    print(f"RMSE: {RMSE}")

    # plt.scatter(X_train, y_train)
    # plt.plot(X_train, y_pred)
    # plt.show()


def main():
    regressions()


if __name__ == "__main__":
    main()
