# random Forest experiments

import csv
from random import randrange

import numpy as np
import sklearn.tree as tree
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, zero_one_loss, precision_score, multilabel_confusion_matrix, \
    confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics
import sklearn.ensemble as ensemble
import time


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
    plt.title("random forest")
    for first_index in range(len(mcm)):
        for second_index in range(len(mcm[first_index])):
            plt.text(second_index, first_index, mcm[first_index][second_index])
    plt.show()


df = pd.read_csv('../Data/final_data_test.csv')
y = df.iloc[:, 50]
x = df.iloc[:, 0: 50]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12345)

# param_grid to test different combination of parameters. 8, 15, 0.3, 4 are selected as the best
# try difference parameters if you want
param_grid = {
    'max_depth': [8],  # 深度：这里是森林中每棵决策树的深度
    'n_estimators': [15],  # 决策树个数-随机森林特有参数
    'max_features': [0.3],  # 每棵决策树使用的变量占比-随机森林特有参数（结合原理）
    'min_samples_split': [4]  # 叶子的最小拆分样本量
}
rfc = ensemble.RandomForestClassifier()
rfc_cv = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=4)
rfc_cv.fit(X_train, y_train)

GridSearchCV(cv=4, estimator=RandomForestClassifier(),
             param_grid={
                 'max_depth': [8],
                 'max_features': [0.3],
                 'min_samples_split': [4],
                 'n_estimators': [15]},
             )

# test tht time
start = time.time()
test_est = rfc_cv.predict(X_test)
end = time.time()
print(end - start)

print(metrics.classification_report(test_est, np.array(y_test), digits=3))
print(calculate_metrics(test_est, np.array(y_test)))
print(rfc_cv.best_params_)
print(rfc_cv.best_score_)
print(rfc_cv.cv_results_['params'])
