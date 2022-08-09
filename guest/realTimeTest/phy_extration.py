# In this code, our aim is to extract data in PHY layer chronograph
# The extraction process is almost the same as MAC layer, The data in MCS should be converted into MCS_index_table
# In MAC layer, we could say lcid number can represent the service type, but here we cannot.
# So the code here only do the data extraction and processing, without further model-testing.


import influxdb_client, os, time
# from influxdb_client import InfluxDBClient, Point, WritePrecision
# from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb import InfluxDBClient
from influxdb import DataFrameClient

import pandas as pd


client = DataFrameClient(host='137.194.194.37', port=8086, username='admin', password='admin')

# dci_sniffer/autogen bucket
bucket = "dci_sniffer/autogen"

query = 'select * from "dci_sniffer"."autogen"."dci_sensor" where time > 1656929885s and time < 1656929886s'
tables = client.query(query)

data = tables['dci_sensor']
# sql_data = pd.DataFrame(tables)
if data.shape[0] < 10:
    print("little information")
elif data.shape[0] > 50:
    data = data.iloc[0: 50]
data_time = data.index
data['time'] = data_time
data.index = range(0, 50)

for row in range(0, 49):
    data.loc[row, 'time'] = (data.loc[row + 1, 'time'] - data.loc[row, 'time']).total_seconds()
    # print((data.loc[row + 1, 'time'] - data.loc[row, 'time']))
data['time'] = data['time'].shift(1)
data.loc[0, 'time'] = 0

tableMCS1 = pd.read_csv('MCS_index_table_1.csv')
tableMCS2 = pd.read_csv('MCS_index_table_2.csv')
tableMCS3 = pd.read_csv('MCS_index_table_3.csv')

conv = lambda x: (x // 51) + 1
data['riv'] = data['riv'].apply(conv)
data['R'] = data['mcs']
data['Qm'] = data['mcs']

# query table to get R and Qm
data['R'] = data['R'].replace(dict(zip(tableMCS2.index, tableMCS2.R)))
data['Qm'] = data['Qm'].replace(dict(zip(tableMCS2.index, tableMCS2.Qm)))
data['NInfo'] = data['Qm'] * data['R'] * data['riv']

print(data)
# print(data)
