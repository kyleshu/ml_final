import data_processing as dp
import pandas as pd
import linear_regression as lr
import matplotlib.pyplot as plt
import itertools
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
import sys
import numpy as np

if __name__ == '__main__':

    conn = dp.import_data_to_db('0714')
    dp.import_data_to_db('full', conn)

    old = sys.stdout
    f = open('results/5-day-prediction.txt', 'w')
    sys.stdout = f

    sql2 = 'select days_since_0122, confirmed from covid_19_data_full where country = \'Italy\' ' \
           'and days_since_0122 between 175 and 179 order by days_since_0122 asc'
    df2 = pd.read_sql(sql2, conn)
    data2 = df2.confirmed.values

    sql3 = 'select days_since_0122, deaths from covid_19_data_full where country = \'Italy\' ' \
           'and days_since_0122 between 175 and 179 order by days_since_0122 asc'
    df3 = pd.read_sql(sql3, conn)
    data3 = df3.deaths.values

    sql4 = 'select days_since_0122, recovered from covid_19_data_full where country = \'Italy\' ' \
           'and days_since_0122 between 175 and 179 order by days_since_0122 asc'
    df4 = pd.read_sql(sql4, conn)
    data4 = df4.recovered.values

    # Linear Regression Confirmed
    sql = 'select days_since_0122, confirmed from covid_19_data_0714 where country = \'Italy\' ' \
          'and days_since_0122 > 140 order by days_since_0122 asc'
    df = pd.read_sql(sql, conn)
    data = df.confirmed.values

    X_train, X_test, Y_train, Y_test, X_lately = lr.prepare_data(data, 5, 0.2)
    score = lr.linear_regression(X_train, X_test, Y_train, Y_test, X_lately, 'confirmed-5', True)

    print("Actual for", 'confirmed-5', ':', data2)

    # Linear Regression Deaths
    sql = 'select days_since_0122, deaths from covid_19_data_0714 where country = \'Italy\' ' \
          'and days_since_0122 > 140 order by days_since_0122 asc'
    df = pd.read_sql(sql, conn)
    data = df.deaths.values

    X_train, X_test, Y_train, Y_test, X_lately = lr.prepare_data(data, 5, 0.2)
    score = lr.linear_regression(X_train, X_test, Y_train, Y_test, X_lately, 'deaths-5', True)

    print("Actual for", 'deaths-5', ':', data3)

    # Linear Regression Recovered
    sql = 'select days_since_0122, recovered from covid_19_data_0714 where country = \'Italy\' ' \
          'and days_since_0122 > 140 order by days_since_0122 asc'
    df = pd.read_sql(sql, conn)
    data = df.recovered.values

    X_train, X_test, Y_train, Y_test, X_lately = lr.prepare_data(data, 5, 0.2)
    model = lr.linear_regression(X_train, X_test, Y_train, Y_test, X_lately, 'recovered-5', True)

    print("Actual for", 'recovered-5', ':', data4)

    # ARIMA
    data_types = ['confirmed', 'deaths', 'recovered']
    actual_data = {'confirmed': data2, 'deaths': data3, 'recovered': data4}
    for data_type in data_types:
        sql = 'select days_since_0122, ' + data_type + ' from covid_19_data_0714 where country = \'Italy\' order by ' \
                                                       'days_since_0122 asc '
        df = pd.read_sql(sql, conn)
        del df['days_since_0122']

        # Define the p, d and q parameters to take any value between 0 and 3
        p = d = q = range(0, 3)
        # Generate all different combinations of p, d and q
        pdq = list(itertools.product(p, d, q))

        aic = []
        parameters = []
        for param in pdq:
            # for param in pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(df, order=param, enforce_stationarity=True, enforce_invertibility=True)

                results = mod.fit()
                # save results in lists
                aic.append(results.aic)
                parameters.append(param)
            except np.linalg.LinAlgError:
                continue
        # find lowest aic
        index_min = min(range(len(aic)), key=aic.__getitem__)

        model = ARIMA(df, order=parameters[index_min])
        model_fit = model.fit(disp=0)

        forecast, stderr, conf_int = model_fit.forecast(steps=5)
        model_fit.plot_predict(start=2, end=len(df) + 4)
        plt.title('arima: ' + data_type)
        plt.show()

        print("Forecast for", "ARIMA-" + data_type, ':', forecast)
        print("Standard error for", "ARIMA-" + data_type, ':', stderr)
        print("95% confidence interval for", "ARIMA-" + data_type, ':', conf_int)
        print("Actual for", data_type, ':', actual_data[data_type])

    sys.stdout = old
    f.close()
