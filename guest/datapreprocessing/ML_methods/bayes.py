# Gaussian bayes experiments

import matplotlib.pyplot
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, zero_one_loss, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import time


df = pd.read_csv('../Data/final_data_test_6_tuan.csv')

acc = []
time_cost = []
# first 50 rows are the data, left is the label
# first 37267 columns are single traffic, the same as in final_data_test_6
# the whole data is mixed traffic
y = df.iloc[:37267, 50: 51]
x = df.iloc[:37267, 0: 50]
# x = df.iloc[:, 50: 53]
# x['max_val'] = x.max(axis=1)
# x['min_val'] = x.min(axis=1)
# x['mean_val'] = x.mean(axis=1)

# split the data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12345)

# Gaussian models
gaussianNB = GaussianNB()
gaussianNB.fit(X_train, y_train)

# time testing
start = time.time()
pred = gaussianNB.predict(X_test)
end = time.time()
print(format(end - start, '.4f'))
print(metrics.classification_report(pred, np.array(y_test), digits=3))

# calculate the classification result
def calculate_metrics(pred, target, threshold=0.5):
    # pred = np.array(pred > threshold, dtype=float)
    # t = classification_report(target, pred, target_names=['HTTP', 'VOIP', 'RTP'])
    print('accuracy: {}'.format(
        accuracy_score(target, pred)))
    # zero_one_loss(target, pred)))
    # precision_score(target, pred, average='samples')))
    mcm = confusion_matrix(target, pred)
    print(mcm)
    plt.imshow(mcm, cmap=plt.cm.Blues)
    indices = range(len(mcm))
    classes = [1, 2, 3]
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.xlabel('guess')
    plt.ylabel('fact')
    plt.title("bayes")
    for first_index in range(len(mcm)):
        for second_index in range(len(mcm[first_index])):
            plt.text(second_index, first_index, mcm[first_index][second_index])
    plt.show()

# print(calculate_metrics(pred, np.array(y_test)))


# testing of the choice of k
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(range(10, 51), acc, label='accuracy', color='blue')
# plt.legend(loc=0)
# ax2 = ax1.twinx()
# ax2.plot(range(10, 51), time_cost, label='OT/s', color='red')
# plt.legend(loc=1)
# ax1.set_xlabel("k")
# plt.title("Experiment Over k")
# plt.show()
