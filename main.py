import data_processing as dp
import model
import numpy as np


if __name__ == '__main__':
    conn = dp.import_data_to_db('full')
    cursor = conn.cursor()
    cursor = cursor.execute('select confirmed from covid_19_data_full where country = \'Italy\' order by confirmed asc')
    data = cursor.fetchall()
    data_array = np.array(data)[:, 0]
    X_train, X_test, Y_train, Y_test, X_lately = model.prepare_data(data_array, 5, 0.2)
    model.linear_regression(X_train, X_test, Y_train, Y_test , X_lately)
