# mixed traffic test for Gaussian Bayes
# use data from final_data_test_6

import numpy as np
from scipy import sparse
from sklearn.multioutput import ClassifierChain
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, hamming_loss, precision_score, confusion_matrix
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('../Data/final_data_test_6.csv')
y = df.iloc[:, 51: 54].values
x = df.iloc[:, 0: 50].values
df = df.iloc[0: 37428]
y = df.iloc[:, 51: 54]
x = df.iloc[:, 0: 50]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=12345)
y_train = y_train.astype(np.float64)
y_test = y_test.astype(np.float64)

# Three types of conversion method to multi-label situation.
classifier = BinaryRelevance(classifier=GaussianNB())
# classifier = ClassifierChain(GaussianNB())
# classifier = GaussianNB()
# classifier = LabelPowerset(GaussianNB())
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
print('accuracy: {}, hamming_loss: {}, precision_score: {}'.format(
                         accuracy_score(np.array(y_test), pred),
                         hamming_loss(np.array(y_test), pred),
                         precision_score(np.array(y_test), pred,
                                         average='samples')))
print(metrics.classification_report(np.array(y_test), pred, digits=3))

# show confusion metrix

# mcm = multilabel_confusion_matrix(np.array(y_test), pred)
# print(mcm)
# mcm[0], mcm[1] and so on, Three metrics
# mcm = mcm[0]
# plt.imshow(mcm, cmap=plt.cm.Blues)
# indices = range(len(mcm))
# classes = [1, 2]
# plt.xticks(indices, classes)
# plt.yticks(indices, classes)
# # plt.colorbar()
# plt.xlabel('guess')
# plt.ylabel('fact')
# plt.title("knn")
# print(mcm[1, 0])
# for first_index in range(len(mcm)):
#     for second_index in range(len(mcm[first_index])):
#         plt.text(second_index, first_index, mcm[first_index, second_index])
# plt.show()
