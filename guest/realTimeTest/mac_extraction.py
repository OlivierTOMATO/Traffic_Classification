# The code is to get mac data (time and l) from chronograph
# The data will be preprocessed and then stored into a CSV file, which is used to train the model.

from influxdb import DataFrameClient

import pandas as pd


def query(num):

    client = DataFrameClient(host='137.194.194.37', port=8086, username='admin', password='admin')

    bucket = "dci_sniffer/autogen"

    # influxQL language to query the database, make modification here to realise further functionalities
    query = 'select * from "dci_sniffer"."autogen"."sdu_sensor" where time > now() - 20d and time < now() - 14d group by lcid'
    # query = 'select l from "dci_sniffer"."autogen"."sdu_sensor" where time > 1656929880s and time < 1656929890s group by lcid'

    # query = "select l from \"dci_sniffer\".\"autogen\".\"sdu_sensor\" where time > '2022-07-04T08:08:55Z' and time < '2022-07-05T08:08:55Z' group by lcid"
    # query = "select l from \"dci_sniffer\".\"autogen\".\"sdu_sensor\" where time > '2022-07-04T08:08:55Z' and time < '2022-07-04T08:09:00Z' group by lcid"
    tables = client.query(query)

    # take out the data in lcid(num), which represents the service type.
    data = tables[('sdu_sensor', (('lcid', num),))]
    data_time = data.index
    data['time'] = data_time
    data.index = range(0, data.shape[0])

    sample_group = []
    contents = []
    time_temp = data.loc[0, 'time']
    for row in data.index:
        if (data.loc[row, 'time'] - time_temp).total_seconds() > 0.5:
            time_temp = data.loc[row, 'time']
            if len(sample_group) > 10:
                if len(sample_group) < 50:
                    sample_group.extend([0] * (50 - len(sample_group)))
                sample_group = sample_group[0: 50]
                sample_group.append(int(num))
                contents.append(sample_group)
            sample_group = []
        else:
            sample_group.append(data.loc[row, 'l'])
    return contents

# save the file into csv for training in advance
def save_csv(arr, csv_filename=None):
    """Save the data in csv format"""
    if csv_filename is None:
        csv_filename = "data_mac_727.csv"
    arr_df = pd.DataFrame(arr)
    arr_df.to_csv(csv_filename, float_format='%.3f', index=False, header=False)

# print(data)
# 5 means RTP, 6 means HTTP
content1 = query('5')
content2 = query('6')
# contents = content1
contents = content1 + content2

save_csv(contents)
