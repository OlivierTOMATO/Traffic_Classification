# use the data in data_mac_727 to train the model. Here, the model is gaussian model.

import joblib
from sklearn.naive_bayes import GaussianNB
import pandas as pd


df = pd.read_csv('data_mac_727.csv')

y = df.iloc[:, 50: 51]
x = df.iloc[:, 0: 50]

gaussianNB = GaussianNB()
gaussianNB.fit(x.values, y.values)
joblib.dump(gaussianNB, 'model/GaussianNB.model')