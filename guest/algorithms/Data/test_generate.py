# code for generating the data for models from the raw data in HTTP.csv and so on
# the aimed data could be csv files for ML methods and npz files for NN methods.
# different combination of features could be selected 3F(Nrb, QM, R) && IAT && TBS(Ninfo)
# it will take 10-20 min to generate all the training data, since there is more than 50k columns of data

"""# IMPORT Data"""

import numpy as np
# load tables
import pandas as pd


# function to generate data
# path is the reference of raw data, num is the expected labeling output of the data
# the function will return a list of data.
# for CSV, NPZ file, the returned data is totally different.
def data_pro(path, num):
    # load the path, read raw data in csv file
    tableData = pd.read_csv(path)
    tableData.describe()
    # drop some unnecessary rows in the raw file
    tableData = tableData.drop(['rv_idx1', 'harq_process'], axis=1)
    tableData.head()
    # load the MCS table
    tableMCS1 = pd.read_csv('MCS_index_table_1.csv')
    tableMCS2 = pd.read_csv('MCS_index_table_2.csv')
    tableMCS3 = pd.read_csv('MCS_index_table_3.csv')

    # insert Qm, R and session, where on session means 50 DCIs transferred in the window size.
    # The DCIs in the same window size shares the same session number.
    tableData.insert(0, 'Qm', '')
    tableData.insert(0, 'R', '')
    tableData.insert(0, 'session', '')

    # change Qm, R into MCS of raw data, and do conversion based on the loaded MCS table respectively
    tableData.R = tableData.mcs
    tableData.Qm = tableData.mcs
    tableData['R'] = tableData['R'].replace(dict(zip(tableMCS2.index, tableMCS2.R)))
    tableData['Qm'] = tableData['Qm'].replace(dict(zip(tableMCS2.index, tableMCS2.Qm)))

    # calculate the number of RB
    b16 = lambda x: int(x, 16)
    tableData['rb_alloc'] = tableData['rb_alloc'].apply(b16)
    convert = lambda x: (x // 51) + 1
    tableData['rb_alloc'] = tableData['rb_alloc'].apply(convert)

    # set the legal of using data, min_length means the least DCIs in a session, max_length means the most one.
    # change different choices of min_length and max_length, and compare the difference.
    min_length = 10
    max_length = 50
    df1 = tableData
    # create a timestamp column to represent the exact time, counted later
    df1.loc[0, 'timestamp'] = df1.loc[0, 'slot'] * 0.5
    df1.loc[0, 'session'] = 1
    time_temp = df1.loc[0, 'timestamp']
    rnti_temp = df1.loc[0, 'rnti']
    session = 1
    # contents to store all the generated data from a csv
    contents = []
    # sample_group to temporarily store the generated data in a session
    sample_group = []
    first = 1

    # start to generate
    for indxrow in df1.index:

        # generate the 3F + IAT (4 features combination)
        # remember that the process here is to generate npz file, all the data point here is stored in a list format
        # within a []
        # if(len(sample_group) < max_length):
        #     if indxrow == 0:
        #         sample_group.append([df1.loc[indxrow, 'R'] / 500, df1.loc[indxrow, 'Qm'], df1.loc[indxrow, 'rb_alloc']
        #         , 0])
        #     else:
        #         sample_group.append([df1.loc[indxrow, 'R'] / 500, df1.loc[indxrow, 'Qm'], df1.loc[indxrow, 'rb_alloc']
        #         , df1.loc[indxrow, 'timestamp'] - df1.loc[indxrow - 1, 'timestamp']])

        # generate the iat (one features combination)
        if len(sample_group) < max_length:
            if indxrow == 0:
                sample_group.append([0])
            else:
                sample_group.append([df1.loc[indxrow, 'timestamp'] - df1.loc[indxrow - 1, 'timestamp']])

        if indxrow == len(df1) - 1:
            break
        # count the timestamp according to frame number and slot number.
        # one frame is 10 ms, and one slot is 0.5 ms, one frame = 20 slots.
        if rnti_temp == df1.loc[indxrow + 1, 'rnti']:
            df1.loc[indxrow + 1, 'timestamp'] = df1.loc[indxrow, 'timestamp'] + (df1.loc[indxrow + 1, 'frame'] -
                                                                                 df1.loc[
                                                                                     indxrow, 'frame']) * 10 + 0.5 * (
                                                        df1.loc[indxrow + 1, 'slot'] - df1.loc[indxrow, 'slot'])
            # refresh of frame number, since when frame number exceeds 1024, it turns to 0.
            if (df1.loc[indxrow + 1, 'frame'] < df1.loc[indxrow, 'frame']) and (indxrow + 1 > indxrow):
                df1.loc[indxrow + 1, 'timestamp'] = df1.loc[indxrow, 'timestamp'] + (
                        1024 + df1.loc[indxrow + 1, 'frame'] -
                        df1.loc[indxrow, 'frame']) * 10 + 0.5 * (df1.loc[indxrow + 1, 'slot']
                                                                 - df1.loc[indxrow, 'slot'])

            # this is the data to generate csv function, noting that all the data generated is stored as a number and
            # then added into the sample_group list, which is different from npz generation.

            # generate the iat (one features combination)
            # if indxrow >= 2 and first == 0:
            # sample_group.append(df1.loc[indxrow + 1, 'timestamp'] - df1.loc[indxrow, 'timestamp'])
            first = 0

            # generate the TBS (one features combination)
            # sample_group.append(df1.loc[indxrow, 'R'] * df1.loc[indxrow, 'Qm'] * df1.loc[indxrow, 'rb_alloc'])

            # trim the generated data, the window size here is 500 ms
            # during the 500 ms, if number of DCIs processed is less than min_length, drop it.
            # if more than 50, trim to 50. Otherwise, do zero padding.
            if df1.loc[indxrow + 1, 'timestamp'] > time_temp + 500 or df1.loc[indxrow + 1, 'timestamp'] < df1.loc[
                indxrow, 'timestamp']:
                time_temp = df1.loc[indxrow + 1, 'timestamp']
                session += 1
                # interval = 500 / len(sample_group)
                if len(sample_group) > min_length:
                    if len(sample_group) < max_length:
                        # zero padding of npz file. pad in [] list file
                        # sample_group.extend([[0, 0, 0, 0]] * (max_length - len(sample_group))) ##four
                        # sample_group.extend([[0, 0]] * (max_length - len(sample_group))) ##two
                        sample_group.extend([[0]] * (max_length - len(sample_group)))

                    # for csv file, one output (when the expected output is 1, 2, 3, 4, 5, 6)
                    # sample_group = sample_group[0: max_length]
                    # sample_group.append(interval)
                    # sample_group.append(num)

                    # for csv file, multiple output
                    # (when the expected output is [1, 0, 0], [0, 1, 0]..., add respectively)
                    # sample_group.append(num[0])
                    # sample_group.append(num[1])
                    # sample_group.append(num[2])
                    # contents.append(sample_group)

                    # for npz file, add directly
                    contents.append((sample_group, num))
                    first = 1
                sample_group = []
            df1.loc[indxrow + 1, 'session'] = session
        else:
            rnti_temp = df1.loc[indxrow + 1, 'rnti']
            df1.loc[indxrow + 1, 'timestamp'] = df1.loc[0, 'slot'] * 0.5
            time_temp = df1.loc[indxrow + 1, 'timestamp']
    return contents


# generate data with labeling 1, 2, 3, 4, 5, 6
# content_1 = data_pro('HTTP.csv', 1)
# content_3 = data_pro('HTTP2.csv', 1)
# content_2 = data_pro('VOIP.csv', 2)
# content_4 = data_pro('VOIP2.csv', 2)
# content_5 = data_pro('RTP2.csv', 3)
# content_6 = data_pro('VOIP+HTTP.csv', 4)
# content_7 = data_pro('RTP+HTTP.csv', 5)
# content_8 = data_pro('RTP+VOIP.csv', 6)

# generate data with labeling [1, 0, 0], [0, 1, 0]...
content_1 = data_pro('HTTP.csv', [1, 0, 0])
content_3 = data_pro('HTTP2.csv', [1, 0, 0])
content_2 = data_pro('VOIP.csv', [0, 1, 0])
content_4 = data_pro('VOIP2.csv', [0, 1, 0])
content_5 = data_pro('RTP2.csv', [0, 0, 1])
content_6 = data_pro('VOIP+HTTP.csv', [1, 1, 0])
content_7 = data_pro('RTP+HTTP.csv', [1, 0, 1])
content_8 = data_pro('RTP+VOIP.csv', [0, 1, 1])

# for single traffic
# contents = content_1 + content_2 + content_3 + content_4 + content_5
# for mixed traffic
contents = content_1 + content_2 + content_3 + content_4 + content_5 + content_6 + content_7 + content_8


# function to generate CSV file, input is the contents. the result will be stored in csv_filename.
def save_csv(arr, csv_filename=None):
    """Save the data in csv format"""
    if csv_filename is None:
        csv_filename = "test_sample(time).csv"
    arr_df = pd.DataFrame(arr)
    arr_df.to_csv(csv_filename, float_format='%.3f', index=False, header=False)


save_csv(contents)


# code to generate npz file.
# inp_temp is input, label_temp is the output, which is useful for NNs' training
for i in range(len(contents)):
    inp_temp = np.array(contents[i][0])
    label_temp = contents[i][1]
    np.savez('test_sample(time)/test_sample_' + str(i), a=inp_temp, b=label_temp)

print("generation ending")
