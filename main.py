import data_processing as dp
import model
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    conn = dp.import_data_to_db('full')
    cursor = conn.cursor()

    cursor = cursor.execute('select confirmed from covid_19_data_full where country = \'Italy\' order by confirmed asc')
    data = cursor.fetchall()
    data_array = np.array(data)[:, 0]
    days = []
    scores = []
    for i in range(1, 31):
        X_train, X_test, Y_train, Y_test, X_lately = model.prepare_data(data_array, i, 0.2)
        score = model.linear_regression(X_train, X_test, Y_train, Y_test, X_lately, 'confirmed'+str(i))
        days.append(i)
        scores.append(score)
    plt.plot(days, scores)
    plt.title('confirmed score')
    plt.savefig('results/confirmed_score.jpg')
    plt.show()


    cursor = cursor.execute('select new_confirmed from covid_19_data_full where country = \'Italy\' order by new_confirmed asc')
    data = cursor.fetchall()
    data_array = np.array(data)[:, 0]
    days = []
    scores = []
    for i in range(1, 31):
        X_train, X_test, Y_train, Y_test, X_lately = model.prepare_data(data_array, i, 0.2)
        score = model.linear_regression(X_train, X_test, Y_train, Y_test, X_lately, 'new_confirmed'+str(i))
        days.append(i)
        scores.append(score)
    plt.plot(days, scores)
    plt.title('new confirmed score')
    plt.savefig('results/new_confirmed_score.jpg')
    plt.show()

    cursor = cursor.execute('select deaths from covid_19_data_full where country = \'Italy\' order by deaths asc')
    data = cursor.fetchall()
    data_array = np.array(data)[:, 0]
    days = []
    scores = []
    for i in range(1, 31):
        X_train, X_test, Y_train, Y_test, X_lately = model.prepare_data(data_array, i, 0.2)
        score = model.linear_regression(X_train, X_test, Y_train, Y_test, X_lately, 'deaths'+str(i))
        days.append(i)
        scores.append(score)
    plt.plot(days, scores)
    plt.title('deaths score')
    plt.savefig('results/deaths_score.jpg')
    plt.show()

    cursor = cursor.execute('select new_deaths from covid_19_data_full where country = \'Italy\' order by new_deaths asc')
    data = cursor.fetchall()
    data_array = np.array(data)[:, 0]
    days = []
    scores = []
    for i in range(1, 31):
        X_train, X_test, Y_train, Y_test, X_lately = model.prepare_data(data_array, i, 0.2)
        score = model.linear_regression(X_train, X_test, Y_train, Y_test, X_lately, 'new_deaths'+str(i))
        days.append(i)
        scores.append(score)
    plt.plot(days, scores)
    plt.title('new deaths score')
    plt.savefig('results/new_deaths_score.jpg')
    plt.show()

    cursor = cursor.execute('select recovered from covid_19_data_full where country = \'Italy\' order by recovered asc')
    data = cursor.fetchall()
    data_array = np.array(data)[:, 0]
    days = []
    scores = []
    for i in range(1, 31):
        X_train, X_test, Y_train, Y_test, X_lately = model.prepare_data(data_array, i, 0.2)
        score = model.linear_regression(X_train, X_test, Y_train, Y_test, X_lately, 'recovered'+str(i))
        days.append(i)
        scores.append(score)
    plt.plot(days, scores)
    plt.title('recovered score')
    plt.savefig('results/recovered_score.jpg')
    plt.show()

    cursor = cursor.execute('select new_recovered from covid_19_data_full where country = \'Italy\' order by new_recovered asc')
    data = cursor.fetchall()
    data_array = np.array(data)[:, 0]
    days = []
    scores = []
    for i in range(1, 31):
        X_train, X_test, Y_train, Y_test, X_lately = model.prepare_data(data_array, i, 0.2)
        score = model.linear_regression(X_train, X_test, Y_train, Y_test, X_lately, 'new_recovered'+str(i))
        days.append(i)
        scores.append(score)
    plt.plot(days, scores)
    plt.title('new_recovered score')
    plt.savefig('results/new_recovered_score.jpg')
    plt.show()
