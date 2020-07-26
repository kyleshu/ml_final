import sqlite3
import csv
import datetime


def import_data_to_db(dataset: str):
    con = sqlite3.connect(':memory:')
    cur = con.cursor()
    cur.execute(
        'create table covid_19_data_' + dataset + '(prov char(128), country char(128), latitude decimal(20,10), '
                                                  'longitude decimal(20,10), confirmed decimal(10,2), deaths decimal('
                                                  '10,2), recovered decimal(10,2), days_since_0122 decimal(10,2), '
                                                  'day_of_week integer, is_weekday integer, new_confirmed decimal(10,'
                                                  '2), new_deaths decimal(10,2), new_recovered decimal(10,2))')
    cur.execute('create index pcd on covid_19_data_' + dataset + ' (prov, country, days_since_0122)')

    rows = []
    with open('data/' + dataset + '/time_series_covid_19_confirmed.csv', 'r', encoding='utf-8') as source:
        lines = csv.reader(source, delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
        is_first_line = True
        jan22 = datetime.datetime.strptime('2020-01-22', '%Y-%m-%d')
        for line in lines:
            if is_first_line:
                is_first_line = False
            else:
                for i in range(4, len(line)):
                    new_row = line[0:2]
                    new_row.append(float(line[2]))
                    new_row.append(float(line[3]))
                    new_row.append(int(line[i]))
                    new_row.append(None)
                    new_row.append(None)
                    new_row.append(i - 4)
                    this_date = jan22 + datetime.timedelta(days=(i - 4))
                    new_row.append(this_date.weekday())
                    if this_date.weekday() > 4:
                        new_row.append(0)
                    else:
                        new_row.append(1)
                    if i != 4:
                        new_row.append(int(line[i]) - int(line[i - 1]))
                    else:
                        new_row.append(int(line[i]))
                    new_row.append(None)
                    new_row.append(None)
                    rows.append(new_row)
    cur = cur.executemany('insert into covid_19_data_' + dataset + ' values(?,?,?,?,?,?,?,?,?,?,?,?,?)', rows)
    updates = []
    with open('data/' + dataset + '/time_series_covid_19_deaths.csv', 'r', encoding='utf-8') as source:
        lines = csv.reader(source, delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
        is_first_line = True
        for line in lines:
            if is_first_line:
                is_first_line = False
            else:
                for i in range(4, len(line)):
                    update = [int(line[i])]
                    if i != 4:
                        update.append(int(line[i]) - int(line[i - 1]))
                    else:
                        update.append(int(line[i]))
                    update.extend(line[0:2])
                    update.append(i-4)
                    updates.append(update)
    cur = cur.executemany('update covid_19_data_' + dataset + ' set deaths=?, new_deaths=? where prov=? and country=? '
                                                              'and days_since_0122=?', updates)
    updates = []
    with open('data/' + dataset + '/time_series_covid_19_recovered.csv', 'r', encoding='utf-8') as source:
        lines = csv.reader(source, delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
        is_first_line = True
        for line in lines:
            if is_first_line:
                is_first_line = False
            else:
                for i in range(4, len(line)):
                    update = [int(line[i])]
                    if i != 4:
                        update.append(int(line[i]) - int(line[i - 1]))
                    else:
                        update.append(int(line[i]))
                    update.extend(line[0:2])
                    update.append(i-4)
                    updates.append(update)
    cur = cur.executemany('update covid_19_data_' + dataset + ' set recovered=?, new_recovered=? where prov=? and '
                                                              'country=? and days_since_0122=?', updates)
    con.commit()
    return con


if __name__ == '__main__':
    conn = import_data_to_db('full')
    cursor = conn.cursor()
    cursor = cursor.execute('select * from covid_19_data_full')
    for row in cursor:
        print(row)
