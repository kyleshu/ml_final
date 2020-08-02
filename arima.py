import pandas as pd
import matplotlib.pyplot as plt
import itertools
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
import sys
import numpy as np

import data_processing as dp

if __name__ == '__main__':

    conn = dp.import_data_to_db('full')

    data_types = ['confirmed', 'deaths', 'recovered']

    for data_type in data_types:
        old = sys.stdout
        f = open('results/arima/' + data_type + '.txt', 'w')
        sys.stdout = f

        sql = 'select days_since_0122, ' + data_type + ' from covid_19_data_full where country = \'Italy\' order by ' \
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
                # seasonal_param.append(param_seasonal)
                print('ARIMA{} - AIC:{}'.format(param, results.aic))
            except np.linalg.LinAlgError:
                continue
        # find lowest aic
        index_min = min(range(len(aic)), key=aic.__getitem__)

        print('The optimal model is: ARIMA{} -AIC{}'.format(parameters[index_min], aic[index_min]))

        model = ARIMA(df, order=parameters[index_min])
        model_fit = model.fit(disp=0)
        print(model_fit.summary())

        model_fit.plot_predict(start=2, end=len(df) + 12)
        plt.title('arima: ' + data_type)
        plt.savefig('results/arima/' + data_type + '.jpg')
        plt.show()

        sys.stdout = old
        f.close()
