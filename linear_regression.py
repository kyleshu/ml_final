import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
import data_processing as dp


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
                      X_lately: np.ndarray, name: str, save_image: bool = False):
    model = linear_model.LinearRegression()

    model.fit(X_train, Y_train)
    score = model.score(X_test, Y_test)
    print("Score for ", name, " is ", score)

    y_test_predict = model.predict(X_test)

    plt.figure(figsize=(16, 8))
    plt.plot(y_test_predict)
    plt.plot(Y_test)
    if save_image:
        plt.savefig('results/linear_regression/'+name+'.jpg')
    plt.show()

    forecast = model.predict(X_lately)
    if save_image:
        print("Forecast for", name, ':', forecast)
    return score


if __name__ == '__main__':

    conn = dp.import_data_to_db('full')
    cursor = conn.cursor()

    cursor = cursor.execute(
        'select days_since_0122, confirmed from covid_19_data_full where country = \'Italy\' order by days_since_0122 asc')
    data = cursor.fetchall()
    data_array = np.array(data)[:, 1]
    days = []
    scores = []
    for i in range(1, 31):
        X_train, X_test, Y_train, Y_test, X_lately = prepare_data(data_array, i, 0.2)
        score = linear_regression(X_train, X_test, Y_train, Y_test, X_lately, 'confirmed' + str(i))
        days.append(i)
        scores.append(score)
    plt.plot(days, scores)
    plt.title('confirmed score')
    plt.savefig('results/linear_regression/confirmed_score.jpg')
    plt.show()

    cursor = cursor.execute(
        'select days_since_0122, new_confirmed from covid_19_data_full where country = \'Italy\' order by days_since_0122 asc')
    data = cursor.fetchall()
    data_array = np.array(data)[:, 1]
    days = []
    scores = []
    for i in range(1, 31):
        X_train, X_test, Y_train, Y_test, X_lately = prepare_data(data_array, i, 0.2)
        score = linear_regression(X_train, X_test, Y_train, Y_test, X_lately, 'new_confirmed' + str(i))
        days.append(i)
        scores.append(score)
    plt.plot(days, scores)
    plt.title('new confirmed score')
    plt.savefig('results/linear_regression/new_confirmed_score.jpg')
    plt.show()

    cursor = cursor.execute(
        'select days_since_0122, deaths from covid_19_data_full where country = \'Italy\' order by days_since_0122 asc')
    data = cursor.fetchall()
    data_array = np.array(data)[:, 1]
    days = []
    scores = []
    for i in range(1, 31):
        X_train, X_test, Y_train, Y_test, X_lately = prepare_data(data_array, i, 0.2)
        score = linear_regression(X_train, X_test, Y_train, Y_test, X_lately, 'deaths' + str(i))
        days.append(i)
        scores.append(score)
    plt.plot(days, scores)
    plt.title('deaths score')
    plt.savefig('results/linear_regression/deaths_score.jpg')
    plt.show()

    cursor = cursor.execute(
        'select days_since_0122, new_deaths from covid_19_data_full where country = \'Italy\' order by days_since_0122 asc')
    data = cursor.fetchall()
    data_array = np.array(data)[:, 1]
    days = []
    scores = []
    for i in range(1, 31):
        X_train, X_test, Y_train, Y_test, X_lately = prepare_data(data_array, i, 0.2)
        score = linear_regression(X_train, X_test, Y_train, Y_test, X_lately, 'new_deaths' + str(i))
        days.append(i)
        scores.append(score)
    plt.plot(days, scores)
    plt.title('new deaths score')
    plt.savefig('results/linear_regression/new_deaths_score.jpg')
    plt.show()

    cursor = cursor.execute(
        'select days_since_0122, recovered from covid_19_data_full where country = \'Italy\' order by days_since_0122 asc')
    data = cursor.fetchall()
    data_array = np.array(data)[:, 1]
    days = []
    scores = []
    for i in range(1, 31):
        X_train, X_test, Y_train, Y_test, X_lately = prepare_data(data_array, i, 0.2)
        score = linear_regression(X_train, X_test, Y_train, Y_test, X_lately, 'recovered' + str(i))
        days.append(i)
        scores.append(score)
    plt.plot(days, scores)
    plt.title('recovered score')
    plt.savefig('results/linear_regression/recovered_score.jpg')
    plt.show()

    cursor = cursor.execute(
        'select days_since_0122, new_recovered from covid_19_data_full where country = \'Italy\' order by days_since_0122 asc')
    data = cursor.fetchall()
    data_array = np.array(data)[:, 1]
    days = []
    scores = []
    for i in range(1, 31):
        X_train, X_test, Y_train, Y_test, X_lately = prepare_data(data_array, i, 0.2)
        score = linear_regression(X_train, X_test, Y_train, Y_test, X_lately, 'new_recovered' + str(i))
        days.append(i)
        scores.append(score)
    plt.plot(days, scores)
    plt.title('new_recovered score')
    plt.savefig('results/linear_regression/new_recovered_score.jpg')
    plt.show()
