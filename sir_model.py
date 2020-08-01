import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize
import pandas as pd
import data_processing as dp


def sir_model(y, x, beta, gamma):
    S = -beta * y[0] * y[1] / N
    R = gamma * y[1]
    I = -(S + R)
    return S, I, R


def fit_odeint(x, beta, gamma):
    return integrate.odeint(sir_model, (S0, I0, R0), x, args=(beta, gamma))[:, 1]


def fit_odeint_susceptible(x, beta, gamma):
    return integrate.odeint(sir_model, (S0, I0, R0), x, args=(beta, gamma))[:, 0]


def fit_odeint_recovered(x, beta, gamma):
    return integrate.odeint(sir_model, (S0, I0, R0), x, args=(beta, gamma))[:, 2]


if __name__ == '__main__':
    conn = dp.import_data_to_db('full')

    sql = 'select days_since_0122, (confirmed-deaths-recovered) as infected from covid_19_data_full ' \
          'where country = \'Italy\' order by  days_since_0122 asc '
    df = pd.read_sql(sql, conn)

    ydata = df.infected.values[30:].astype(float) / 150000  # 这个30可以调更好地适应模型
    xdata = df.days_since_0122.values[30:].astype(float)  # 这个30可以调更好地适应模型

    x_all = [i for i in range(30, 190)]
    x_all = np.array(x_all)

    N = 1.0
    I0 = ydata[0]
    S0 = N - I0
    R0 = 0.0

    popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata)
    print("Beta:", popt[0])
    print("Gamma:", popt[1])

    fitted = fit_odeint(x_all, *popt)
    fitted2 = fit_odeint_susceptible(x_all, *popt)
    fitted3 = fit_odeint_recovered(x_all, *popt)

    plt.plot(xdata, ydata, 'o')
    plt.plot(x_all, fitted)
    plt.plot(x_all, fitted2)
    plt.plot(x_all, fitted3)
    plt.title('SIR infection model')
    plt.savefig('results/sir/infection.jpg')
    plt.show()
