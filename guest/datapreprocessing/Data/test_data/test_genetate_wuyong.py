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
def data_pro(path, num):
    tableVOIP = pd.read_csv(path)
    # tableVOIP = pd.read_csv('VOIP+HTTP.csv')
    # tableHTTP = pd.read_csv('http_test.csv')
    # tableVOIP = pd.read_csv('voip_test.csv')


    tableVOIP.describe()

    tableVOIP = tableVOIP.drop(['rv_idx1', 'harq_process'], axis=1)
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

    from math import floor

    # calculate the number of RB
    ## what is rb

    convert = lambda x: (x // 51) + 1
    tableVOIP['rb_alloc'] = tableVOIP['rb_alloc'].apply(convert)

    min_length = 10
    max_length = 50
    df1 = tableVOIP
    df1.loc[0, 'timestamp'] = df1.loc[0, 'slot'] * 0.5
    df1.loc[0, 'session'] = 1
    time_temp = df1.loc[0, 'timestamp']
    rnti_temp = df1.loc[0, 'rnti']
    session = 1
    contents = []
    sample_group = []
    k = [0, 0, 0]

    for indxrow in df1.index:
        if(len(sample_group) < max_length):
            if indxrow == 0:
                sample_group.append([df1.loc[indxrow, 'R'] / 500, df1.loc[indxrow, 'Qm'], df1.loc[indxrow, 'rb_alloc'], 0])
            else:
                sample_group.append([df1.loc[indxrow, 'R'] / 500, df1.loc[indxrow, 'Qm'], df1.loc[indxrow, 'rb_alloc'], df1.loc[indxrow, 'timestamp'] - df1.loc[indxrow - 1, 'timestamp']])
        # sample_group.append(df1.loc[indxrow, 'R'] * df1.loc[indxrow, 'Qm'] * df1.loc[indxrow, 'rb_alloc'])
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
                        sample_group.extend([[0, 0, 0, 0]] * (max_length - len(sample_group))) ##four
                        # sample_group.extend([0] * (max_length - len(sample_group)))
                    sample_group = sample_group[0: max_length]
                    # sample_group.append(num[0])
                    # sample_group.append(num[1])
                    # sample_group.append(num[2])
                    # contents.append(sample_group[0:max_length + 3])
                    num = sample_group
                    contents.append((sample_group, num)) ##four
                    # num += 1
                sample_group = []
                k = [0, 0, 0]
            df1.loc[indxrow + 1, 'session'] = session
        else:
            rnti_temp = df1.loc[indxrow + 1, 'rnti']
            df1.loc[indxrow + 1, 'timestamp'] = df1.loc[0, 'slot'] * 0.5
            time_temp = df1.loc[indxrow + 1, 'timestamp']
    return contents

# content_1 = data_pro('HTTP.csv', 1)
# content_3 = data_pro('HTTP2.csv', 1)
# content_2 = data_pro('VOIP.csv', 2)
# content_4 = data_pro('VOIP2.csv', 2)
# content_5 = data_pro('RTP2.csv', 3)
# content_6 = data_pro('VOIP+HTTP.csv', 4)
# content_7 = data_pro('RTP+HTTP.csv', 5)
# content_8 = data_pro('RTP+VOIP.csv', 6)

# content_1 = data_pro('HTTP.csv', [0, 0, 1])
# content_3 = data_pro('HTTP2.csv', [0, 0, 1])
# content_2 = data_pro('VOIP.csv', [0, 1, 0])
# content_4 = data_pro('VOIP2.csv', [0, 1, 0])
# content_5 = data_pro('RTP2.csv', [0, 0, 1])
# content_6 = data_pro('VOIP+HTTP.csv', [1, 1, 0])
# content_7 = data_pro('RTP+HTTP.csv', [1, 0, 1])
# content_8 = data_pro('RTP+VOIP.csv', [0, 1, 1])
# contents = content_1 + content_2 + content_3 + content_4 + content_5 + content_6 + content_7 + content_8
contents = data_pro('multiple_traffic.csv', 1)


def save_csv(arr, csv_filename=None):
    """Save the data in csv format"""
    if csv_filename == None:
        csv_filename="multiple_traffic_tuan.csv"
    arr_df = pd.DataFrame(arr)
    arr_df.to_csv(csv_filename, float_format='%.3f', index=False, header=False)

save_csv(contents)
for i in range(len(contents)):
    inp_temp = np.array(contents[i][0])
    label_temp = contents[i][1]
    np.savez('multiple_traffic/test_sample_' + str(i), a=inp_temp, b=label_temp)

print(1)