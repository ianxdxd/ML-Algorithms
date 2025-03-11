import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from Utils.evaluation_metrics import EvaluationMetrics
from Supervised.regression import LinearRegression


def regressions(df_path: str, validation_df_path: str, target: str):
    # df = pd.read_csv(df_path)
    # y = df[target]
    # X = df.drop(labels=target, axis=1)
    # X = X.reset_index(drop=True)
    # y = y.reset_index(drop=True)

    X, y = make_regression(n_samples=500, n_features=1, noise=15, random_state=4)

    # if validation_df_path:
    #     validation_df = pd.read_csv(validation_df_path)
    #     y_val = validation_df[target]
    #     X_val = validation_df.drop(labels=target, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    lr = LinearRegression(1000, 0.001)
    lr.gradientDescent(X_train, y_train)
    lr.printWeights()
    y_pred = lr.predict(X_train)
    lr.printWeights()

    # np.savetxt("y_test.txt", y_test)
    # print(y_test.shape, y_pred.shape)
    # if np.any(np.isnan(y_test)):
    #     print("NaN values detected in target variable!")
    # if np.any(np.isnan(y_train)):
    #     print("NaN values detected in predictions!")

    MAE = EvaluationMetrics(n=X_train.shape[0], y=y_train, y_pred=y_pred).mae()
    print(f"MAE: {MAE}")
    # MSE = EvaluationMetrics(n=X_test.shape[0], y=y_test, y_pred=y_pred).mse()
    # print(f"MSE: {MSE}")

    plt.scatter(X_train, y_train)
    plt.plot(X_train, y_pred)
    plt.show()


def main():
    regressions("Data/test.csv", "Data/train.csv", "y")


if __name__ == "__main__":
    main()
