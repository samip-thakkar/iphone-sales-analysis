# -*- coding: utf-8 -*-
"""

@author: Samip
"""

#Importing dataset
import pandas as pd
df = pd.read_csv('iphone_purchase_records.csv')
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#Convert Gender to numbers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 0] = le.fit_transform(x[:, 0])

#Splitting the dataset to training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.75, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
x_sc = StandardScaler()
x_train = x_sc.fit_transform(x_train)
x_test = x_sc.fit_transform(x_test)


#Fitting SVM model with Radial Basis Function kernel
from sklearn.svm import SVC
svr = SVC(kernel = "rbf", random_state = 0)
svr.fit(x_train, y_train)


#Making predictions
y_pred = svr.predict(x_test)

#Checking Accuracy for x_test
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ", cm)
acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)
pre = metrics.precision_score(y_test, y_pred)
print("Precision score: ", pre)
rec = metrics.recall_score(y_test, y_pred)
print("Recall score: ", rec)
