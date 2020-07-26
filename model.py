import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt


def prepare_data(column: np.ndarray, forecast_out: int, test_size: float):
    df = pd.DataFrame(column)
    label = df.shift(-forecast_out)
    X = np.array(df)
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]
    label.dropna(inplace=True)
    y = np.array(label)

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size)

    response = [X_train, X_test, Y_train, Y_test, X_lately]
    return response


def linear_regression(X_train: np.ndarray, X_test: np.ndarray, Y_train: np.ndarray, Y_test: np.ndarray,
                      X_lately: np.ndarray):
    model = linear_model.LinearRegression()

    model.fit(X_train, Y_train)
    score = model.score(X_test, Y_test)
    print("Score is ", score)

    y_test_predict = model.predict(X_test)

    plt.figure(figsize=(16, 8))
    plt.plot(y_test_predict)
    plt.plot(Y_test)
    plt.show()

    forecast = model.predict(X_lately)
    print("Forecast is ", forecast)
