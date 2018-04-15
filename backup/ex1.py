# Starting Script
print("Starting Script")

import numpy as np
from sklearn import datasets, linear_model, preprocessing, cross_validation, svm
import pandas
import math
from sklearn.linear_model import LinearRegression

filename = 'insurance.csv'
names = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
df = pandas.read_csv(filename, names=names)

df = df[['age','charges']]

# print(df.head())

forecase_col = 'charges'

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecase_col].shift(-forecast_out)
df.dropna(inplace=True)

print(df['label'])

X = np.array(df.drop(['label'],1))
y = np.array(df['label'])
X = preprocessing.scale(X)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y , test_size=0.2)

clf= LinearRegression()
clf.fit(X_train, y_train)
confidence = clf.score(X_test,y_test)

print(confidence)