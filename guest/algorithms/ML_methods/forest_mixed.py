# mixed traffic test for random forest
# use data from final_data_test_6

import numpy as np
# from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, hamming_loss, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset
import pandas as pd
from sklearn import tree

df = pd.read_csv('../Data/final_data_test_6.csv')
y = df.iloc[:, 51: 54].values
x = df.iloc[:, 0: 50].values
# df = df.iloc[0: 37265]
# y = df.iloc[:, 50]
# x = df.iloc[:, 0: 50]
# df['max_val'] = x.max(axis=1)
# df['min_val'] = x.min(axis=1)
# df['mean_val'] = x.mean(axis=1)
# df['time'] = df.iloc[:, 50]
# x = df.iloc[:, 54: 58]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=12345)
y_train = y_train.astype(np.float64)
y_test = y_test.astype(np.float64)

# Three types of conversion method to multi-label situation.
classifier = BinaryRelevance(
    classifier = RandomForestClassifier(max_depth=8, max_features=0.3, min_samples_split=8, n_estimators=15, criterion='gini'),
    require_dense = [False, True]
)
# classifier = ClassifierChain(RandomForestClassifier(max_depth=8, max_features=0.3, min_samples_split=8, n_estimators=15, criterion='gini'))
# classifier = LabelPowerset(RandomForestClassifier(max_depth=8, max_features=0.3, min_samples_split=8, n_estimators=15, criterion='gini'))
# classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
print('accuracy: {}, hamming_loss: {}, precision_score: {}'.format(
                         accuracy_score(np.array(y_test), pred),
                         hamming_loss(np.array(y_test), pred),
                         precision_score(np.array(y_test), pred,
                                         average='samples')))
print(metrics.classification_report(np.array(y_test), pred, digits=3))