# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:43:04 2019

@author: Samip
"""

#Importing the dataset
import pandas as pd
df = pd.read_csv("iphone_purchase_records.csv")
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#Convert Gender to number
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 0] = le.fit_transform(x[:, 0])


#Split dataset to training and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.75, random_state = 0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
x_sc = StandardScaler()
x_train = x_sc.fit_transform(x_train)
x_test = x_sc.fit_transform(x_test)

#Fitting KNN classifier with L2 norm (Euclideon Distance)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, p = 2)
knn.fit(x_train, y_train)

#Make prediction
y_pred = knn.predict(x_test)

#Calculating accuracy
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)
acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy score: ", acc)
pre = metrics.precision_score(y_test, y_pred)
print("Precision score: ", pre)
rec = metrics.recall_score(y_test, y_pred)
print("Recall score: ", rec)

