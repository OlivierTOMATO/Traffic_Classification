# KNN experiments

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import time
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('../Data/final_data_test.csv')
y = df.iloc[:, 50]
x = df.iloc[:, 0: 50]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12345)

# KNN model loading
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
start = time.time()
pred = clf.predict(X_test)
end = time.time()
print(end - start)


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
    # plt.colorbar()
    plt.xlabel('guess')
    plt.ylabel('fact')
    plt.title("knn")
    print(mcm[1, 0])
    for first_index in range(len(mcm)):
        for second_index in range(len(mcm[first_index])):
            plt.text(second_index, first_index, mcm[first_index, second_index])
    plt.show()


print('knn精确度...')
print(metrics.classification_report(pred, np.array(y_test), digits=3))
print(calculate_metrics(pred, np.array(y_test)))
