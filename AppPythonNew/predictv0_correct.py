# Import libraries
import os
from joblib import load
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score

PATH = os.getcwd()

clf = load(PATH + '/pickle/model.pkl')

data = pd.read_csv(PATH+"/data/input/test_cases.csv")

y = data['y']

data.drop('y', axis=1, inplace=True)

data_21 = data.iloc[:, 1:21]
y_pred = clf.predict(data_21)

data["y_pred"] = y_pred

print(data.columns)

data.columns

print(data.head())
print("Accuracy = ", accuracy_score(y, y_pred))
print("Recall   = ", recall_score(y, y_pred, pos_label='yes'))