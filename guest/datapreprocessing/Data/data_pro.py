import time

"""# IMPORT Data"""

import sklearn
# load tables
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
def data_pro():
    tableVOIP = pd.read_csv('VOIP.csv')
    tableHTTP = pd.read_csv('HTTP.csv')
    # tableHTTP = pd.read_csv('http_test.csv')
    # tableVOIP = pd.read_csv('voip_test.csv')


    tableVOIP.describe()

    tableHTTP.tail()

    tableVOIP = tableVOIP.drop(['rv_idx1', 'harq_process'], axis=1)
    tableHTTP = tableHTTP.drop(['rv_idx1', 'harq_process'], axis=1)
    tableVOIP.head()

    tableMCS1 = pd.read_csv('MCS_index_table_1.csv')
    tableMCS2 = pd.read_csv('MCS_index_table_2.csv')
    tableMCS3 = pd.read_csv('MCS_index_table_3.csv')

    """VOIP and HTTP"""

    tableVOIP.insert(0, 'Qm', '')
    tableVOIP.insert(0, 'R', '')
    tableVOIP.insert(0, 'session', '')
    tableVOIP.R = tableVOIP.mcs
    tableVOIP.Qm = tableVOIP.mcs

    tableVOIP['R'] = tableVOIP['R'].replace(dict(zip(tableMCS2.index, tableMCS2.R)))
    tableVOIP['Qm'] = tableVOIP['Qm'].replace(dict(zip(tableMCS2.index, tableMCS2.Qm)))
    b16 = lambda x: int(x, 16)
    tableVOIP['rb_alloc'] = tableVOIP['rb_alloc'].apply(b16)

    tableHTTP.insert(0, 'Qm', '')
    tableHTTP.insert(0, 'R', '')
    tableHTTP.insert(0, 'session', '')
    tableHTTP.R = tableHTTP.mcs
    tableHTTP.Qm = tableHTTP.mcs

    tableHTTP['R'] = tableHTTP['R'].replace(dict(zip(tableMCS2.index, tableMCS2.R)))
    tableHTTP['Qm'] = tableHTTP['Qm'].replace(dict(zip(tableMCS2.index, tableMCS2.Qm)))
    b16 = lambda x: int(x, 16)
    tableHTTP['rb_alloc'] = tableHTTP['rb_alloc'].apply(b16)

    from math import floor

    # calculate the number of RB
    ## what is rb

    convert = lambda x: (x // 51) + 1
    tableVOIP['rb_alloc'] = tableVOIP['rb_alloc'].apply(convert)
    tableHTTP['rb_alloc'] = tableHTTP['rb_alloc'].apply(convert)

    min_length = 30
    max_length = 50
    df1 = tableVOIP
    df1.loc[0, 'timestamp'] = df1.loc[0, 'slot'] * 0.5
    df1.loc[0, 'session'] = 1
    time_temp = df1.loc[0, 'timestamp']
    rnti_temp = df1.loc[0, 'rnti']
    session = 1
    contents = []
    sample_group = []

    for indxrow in df1.index:
        # sample_group.append(df1.loc[indxrow, ['R', 'Qm', 'rb_alloc']])
        if(len(sample_group) < max_length):
            sample_group.append([df1.loc[indxrow, 'R'], df1.loc[indxrow, 'Qm'], df1.loc[indxrow, 'rb_alloc']])
        if (indxrow == len(df1) - 1):
            break
        if rnti_temp == df1.loc[indxrow + 1 , 'rnti']:
            df1.loc[indxrow + 1, 'timestamp'] = df1.loc[indxrow, 'timestamp'] + (df1.loc[indxrow + 1, 'frame'] -
                                    df1.loc[indxrow, 'frame']) * 10 + 0.5 * (df1.loc[indxrow + 1, 'slot']
                                                                             - df1.loc[indxrow, 'slot'])
            if ((df1.loc[indxrow + 1, 'frame'] < df1.loc[indxrow, 'frame']) and (indxrow + 1 > indxrow)):
                df1.loc[indxrow + 1, 'timestamp'] = df1.loc[indxrow, 'timestamp'] + (1024 + df1.loc[indxrow + 1, 'frame'] -
                                        df1.loc[indxrow, 'frame']) * 10 + 0.5 * (df1.loc[indxrow + 1, 'slot']
                                                                                 - df1.loc[indxrow, 'slot'])
            if (df1.loc[indxrow + 1, 'timestamp'] > time_temp + 500):
                time_temp = df1.loc[indxrow + 1, 'timestamp']
                session += 1
                if (len(sample_group) > min_length):
                    if (len(sample_group) < max_length):
                        sample_group.extend([[0, 0, 0]] * (max_length - len(sample_group)))
                    contents.append((sample_group, 1))
                sample_group = []
            df1.loc[indxrow + 1, 'session'] = session
        else:
            rnti_temp = df1.loc[indxrow + 1, 'rnti']
            df1.loc[indxrow + 1, 'timestamp'] = df1.loc[0, 'slot'] * 0.5
            time_temp = df1.loc[indxrow + 1, 'timestamp']

    df2 = tableHTTP
    df2.loc[0, 'timestamp'] = df2.loc[0, 'slot'] * 0.5
    df2.loc[0, 'session'] = 1
    time_temp = df2.loc[0, 'timestamp']
    rnti_temp = df2.loc[0, 'rnti']
    session = 1
    sample_group = []

    for indxrow in df2.index:
        # sample_group.append(df1.loc[indxrow, ['R', 'Qm', 'rb_alloc']])
        if(len(sample_group) <= max_length):
            sample_group.append([df2.loc[indxrow, 'R'], df2.loc[indxrow, 'Qm'], df2.loc[indxrow, 'rb_alloc']])
        if (indxrow == len(df2) - 1):
            break
        if rnti_temp == df2.loc[indxrow + 1 , 'rnti']:
            df2.loc[indxrow + 1, 'timestamp'] = df2.loc[indxrow, 'timestamp'] + (df2.loc[indxrow + 1, 'frame'] -
                                    df2.loc[indxrow, 'frame']) * 10 + 0.5 * (df2.loc[indxrow + 1, 'slot']
                                                                             - df2.loc[indxrow, 'slot'])
            if ((df2.loc[indxrow + 1, 'frame'] < df2.loc[indxrow, 'frame']) and (indxrow + 1 > indxrow)):
                df2.loc[indxrow + 1, 'timestamp'] = df2.loc[indxrow, 'timestamp'] + (1024 + df1.loc[indxrow + 1, 'frame'] -
                                        df2.loc[indxrow, 'frame']) * 10 + 0.5 * (df2.loc[indxrow + 1, 'slot']
                                                                                 - df2.loc[indxrow, 'slot'])
            if (df2.loc[indxrow + 1, 'timestamp'] > time_temp + 500):
                time_temp = df2.loc[indxrow + 1, 'timestamp']
                session += 1
                if (len(sample_group) > min_length):
                    if (len(sample_group) < max_length):
                        sample_group.extend([[0, 0, 0]] * (max_length - len(sample_group)))
                    if (len(sample_group) == 51):
                        sample_group = sample_group[0:50]
                    contents.append((sample_group, 2))
                sample_group = []
            df2.loc[indxrow + 1, 'session'] = session
        else:
            rnti_temp = df2.loc[indxrow + 1, 'rnti']
            df2.loc[indxrow + 1, 'timestamp'] = df2.loc[0, 'slot'] * 0.5
            time_temp = df2.loc[indxrow + 1, 'timestamp']
    for i in range(len(contents)):
        inp_temp = np.array(contents[i][0])
        label_temp = np.array(contents[i][1])
        np.savez('data_sample_2/train_sample_'+str(i), a=inp_temp, b=label_temp)
data_pro()
print(1)