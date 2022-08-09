# Experiments over real-time testing.
# noting that it was not absolutely real-time, it starts from 1656922135s(2022-07-04 16:08:55) and do 'real-time' test
# To change the start-time or alter it to 'now()', please modify the query

from datetime import datetime

import influxdb_client, os, time
import joblib
from influxdb import DataFrameClient

import pandas as pd


# given start_time and end_time and num (lcid num), return the query result and preprocessing data.
# during the process, we take 50 data points during given period of time.
# if the data points is less than 50, fill in zero points.
# the return data is 50 + 1, 50 is the data_points, 1 is the lcid num.
def query(num, start, end):
    # host is the ip address of db
    client = DataFrameClient(host='137.194.194.37', port=8086, username='admin', password='admin')

    # bucket defined in influx_db
    bucket = "dci_sniffer/autogen"

    # influxQL language to query the database, make modification here to realise further functionalities
    query = 'select l from "dci_sniffer"."autogen"."sdu_sensor" where time >' + str(start) + 's and time <' + str(
        end) + 's group by lcid'
    # query = "select l from \"dci_sniffer\".\"autogen\".\"sdu_sensor\" where time >" +'2022-07-04T08:08:55Z' and
    # time < '2022-07-04T08:09:00Z' group by lcid"

    # identify whether there is data detected, otherwise len(tables) should be 0
    tables = client.query(query)
    if len(tables) <= 0:
        return

    # take out the data in lcid(num)
    data = tables[('sdu_sensor', (('lcid', num),))]

    if len(data) == 0 or data is None:
        return
    data_time = data.index
    data['time'] = data_time
    data.index = range(0, data.shape[0])

    sample_group = []
    contents = []
    time_temp = data.loc[0, 'time']
    for row in data.index:
        sample_group.append(data.loc[row, 'l'])
    if len(sample_group) < 10:
        return
    if len(sample_group) < 50:
        sample_group.extend([0] * (50 - len(sample_group)))
    sample_group = sample_group[0: 50]
    sample_group.append(int(num))
    contents.append(sample_group)
    return contents


# load the model we want, here it was the gaussian model.
lr = joblib.load('model/GaussianNB.model')

# keep the query run to realize real-time
start = 1656922135  # start time initialization
end = 1656922136  # end time initialization (here, time window is 1s, means one prediction every second)
# start = math.ceil(time.time()) - 10
# end = math.ceil(time.time())
while True:
    print(datetime.fromtimestamp(start))  # real-time
    content1 = query('5', start, end)  # preprocessed data
    if content1 is not None and len(content1) != 0:
        df = pd.DataFrame(content1)
        y = df.iloc[:, 50: 51]
        x = df.iloc[:, 0: 50]

        pred = lr.predict(x)  # prediction based on the Gaussian model.
        # print(metrics.classification_report(pred, np.array(y), digits=3))
        # print(metrics.accuracy_score(pred, np.array(y)))
        print(pred)

    start = end
    # end = math.ceil(time.time() - 1978508)
    end = start + 1
    time.sleep(1)
