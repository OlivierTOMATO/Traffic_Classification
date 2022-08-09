# visualization of the data on their TBS and IAT.
# counting the TBS and IAT, plot them in a file.
# counting the digital features (mean, max, min, var) of TBS and IAT
# apply elbow method below to see the difference. Description of elbow methods see reference my report (for k choice)

import time

"""# IMPORT Data"""

import sklearn
# load tables

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.fftpack import fft, ifft
import math

from matplotlib import rc
from pandas.plotting import register_matplotlib_converters


# load the data, almost the same as the data in test_generate.
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

    tableVOIP.insert(0, 'Qm', '')
    tableVOIP.insert(0, 'R', '')
    tableVOIP.insert(0, 'session', '')
    tableVOIP.R = tableVOIP.mcs
    tableVOIP.Qm = tableVOIP.mcs

    tableVOIP['R'] = tableVOIP['R'].replace(dict(zip(tableMCS2.index, tableMCS2.R)))
    tableVOIP['Qm'] = tableVOIP['Qm'].replace(dict(zip(tableMCS2.index, tableMCS2.Qm)))
    b16 = lambda x: int(x, 16)
    tableVOIP['rb_alloc'] = tableVOIP['rb_alloc'].apply(b16)

    convert = lambda x: (x // 51) + 1
    tableVOIP['rb_alloc'] = tableVOIP['rb_alloc'].apply(convert)

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
        if (indxrow == len(df1) - 1):
            break
        if rnti_temp == df1.loc[indxrow + 1, 'rnti']:
            df1.loc[indxrow + 1, 'timestamp'] = df1.loc[indxrow, 'timestamp'] + (df1.loc[indxrow + 1, 'frame'] -
                                                                                 df1.loc[
                                                                                     indxrow, 'frame']) * 10 + 0.5 * (
                                                        df1.loc[indxrow + 1, 'slot']
                                                        - df1.loc[indxrow, 'slot'])
            if ((df1.loc[indxrow + 1, 'frame'] < df1.loc[indxrow, 'frame']) and (indxrow + 1 > indxrow)):
                df1.loc[indxrow + 1, 'timestamp'] = df1.loc[indxrow, 'timestamp'] + (
                        1024 + df1.loc[indxrow + 1, 'frame'] -
                        df1.loc[indxrow, 'frame']) * 10 + 0.5 * (df1.loc[indxrow + 1, 'slot']
                                                                 - df1.loc[indxrow, 'slot'])
            if indxrow > 10000:
                break
            if indxrow > 0:
                Ninfo = df1.loc[indxrow, 'R'] / 1024 * df1.loc[indxrow, 'Qm'] * df1.loc[indxrow, 'rb_alloc'] * 84
                TBS_bits = 0
                if Ninfo > 3824:
                    n = math.floor(math.log(Ninfo - 24, 2)) - 5
                    Ninfo_prime = 2 ^ n * round((Ninfo - 24) / (2 ^ n))
                    if df1.loc[indxrow, 'R'] <= 0.25 * 1024:
                        C = math.ceil((Ninfo_prime + 24) / 3816)
                        TBS_bits = 8 * C * math.ceil((Ninfo_prime + 24) / (8 * C)) - 24
                    else:
                        if Ninfo_prime > 8424:
                            C = math.ceil((Ninfo_prime + 24) / 8424)
                            TBS_bits = 8 * C * math.ceil((Ninfo_prime + 24) / (8 * C)) - 24
                        else:
                            TBS_bits = 8 * math.ceil((Ninfo_prime + 24) / 8) - 24
                else:
                    n = max(3, math.floor(math.log(Ninfo, 2)) - 6)
                    Ninfo_prime = max(24, pow(2, n)) * math.floor(Ninfo / pow(2, n))
                    TBS_bits = math.ceil(Ninfo_prime / 2) * 2
                sample_group.append([int(2 * df1.loc[indxrow, 'timestamp']), TBS_bits])
            df1.loc[indxrow + 1, 'session'] = session
        else:
            rnti_temp = df1.loc[indxrow + 1, 'rnti']
            df1.loc[indxrow + 1, 'timestamp'] = df1.loc[0, 'slot'] * 0.5
            time_temp = df1.loc[indxrow + 1, 'timestamp']
    # return contents

    return sample_group


# contents = data_pro('HTTP2.csv', 1)
# k = np.array(contents[25: 391]) ## for http

# contents = data_pro('VOIP2.csv', 1)
# k = np.array(contents[533: 1173])  ## for voip

contents = data_pro('RTP2.csv', 1)
k = np.array(contents[1291: 3247])  ## for rtp

k[0][0] = 0
for i in range(len(k) - 2):
    k[len(k) - i - 1][0] = (k[len(k) - i - 1][0] - k[len(k) - i - 2][0]) / 2


# count the whole digital features of whole 10-s data
TBS_res = []
IAT_res = []
TBS_var_res = []
IAT_var_res = []
mean_TBS_stan = np.mean(k[:, 1])
mean_IAT_stan = np.mean(k[:, 0])
var_TBS_stan = np.var(k[:, 1])
var_IAT_stan = np.var(k[:, 0])

print("TBS_mean: %f" % mean_TBS_stan)
print("TBS_max: %f" % np.max(k[:, 1]))
print("TBS_min: %f" % np.min(k[:, 1]))
print("TBS_mean_variance: %f" % np.var(k[:, 1]))
print("IAT_mean: %f" % np.mean(k[:, 0]))
print("IAT_max: %f" % np.max(k[:, 0]))
print("IAT_min: %f" % np.min(k[:, 0]))
print("IAT_mean_variance: %f" % np.var(k[:, 0]))

# pick from 10 to 500 DCIs every session
# count the digital features and then check the difference from the whole 10-s data respectively. plot the difference
for size in range(10, 500, 10):
    mean_IAT_var = 0
    mean_TBS_var = 0
    var_IAT_var = 0
    var_TBS_var = 0
    count = 0

    for i in range(1, len(k) - 2):
        if i % size == 0:
            mean_TBS_var += pow((np.mean(k[i - size: i, 1]) - mean_TBS_stan), 2)
            mean_IAT_var += pow((np.mean(k[i - size: i, 0]) - mean_IAT_stan), 2)
            var_TBS_var += pow((np.var(k[i - size: i, 1]) - var_TBS_stan), 2)
            var_IAT_var += pow((np.var(k[i - size: i, 0]) - var_IAT_stan), 2)
            print("TBS_mean: %f" % np.mean(k[i: i + size, 1]))
            print("TBS_max: %f" % np.max(k[i: i + size, 1]))
            print("TBS_min: %f" % np.min(k[i: i + size, 1]))
            print("TBS_mean_variance: %f" % np.var(k[i: i + size, 1]))
            print("IAT_mean: %f" % np.mean(k[i: i + size, 0]))
            print("IAT_max: %f" % np.max(k[i: i + size, 0]))
            print("IAT_min: %f" % np.min(k[i: i + size, 0]))
            print("IAT_mean_variance: %f" % np.var(k[i: i + size, 0]))
            count += 1
    TBS_res.append(mean_TBS_var / count)
    IAT_res.append(mean_IAT_var / count)
    TBS_var_res.append(var_TBS_var / count)
    IAT_var_res.append(var_IAT_var / count)
# plt.hist(k[:, 0], bins=100, rwidth=0.5, range=(0, 60))
# plt.title("IAT histogram")
# plt.show()
plt.plot(range(10, 500, 10), TBS_res)
plt.show()
plt.plot(range(10, 500, 10), IAT_res)
plt.show()
plt.plot(range(10, 500, 10), TBS_var_res)
plt.show()
plt.plot(range(10, 500, 10), IAT_var_res)
plt.show()

# plt the TBS and fft of that.
# could be ignored
# def get_window(num, contents):
#     start = 0
#     k = [0] * num * 2
#     t = 0
#     for i, j in contents:
#         if i >= num * 2 + start:
#             break
#         if i > start:
#             t += 1
#             k[i - start] = j
#     return k, t
# k, t =get_window(10000, contents)
# print("Num_DCI: %f" % t)
# fft_k = fft(k)
# a = np.array(range(len(k)))
# a = 2 * ((a - len(k)/2) / 500)
# plt.plot(range(len(k)), k)
# plt.xlabel("time/slot(0.5 ms)")
# plt.ylabel("TBS/bits", loc="top", labelpad=0.5)
# plt.show()
# x = range(-len(k), len(k), 2)
# plt.plot(a, fft_k/500)
# plt.xlabel("frequency/kHz")
# plt.ylabel("amplitude/bits", loc="top", labelpad=-6)
# plt.ylim([-150, 250])
# plt.show()
# visu(contents)
