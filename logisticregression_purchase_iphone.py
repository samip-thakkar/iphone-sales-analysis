# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:23:45 2019

@author: Samip
"""

#Load dataset 
import pandas as pd
df = pd.read_csv('iphone_purchase_records.csv')
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#Convert gender to numbers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 0] = le.fit_transform(x[:, 0])

#Splitting data to training and testing dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.75)

#Feature selection for Logistic Regression as higher values might dominate lower values
from sklearn.preprocessing import StandardScaler
x_sc = StandardScaler()
x_train = x_sc.fit_transform(x_train)
x_test = x_sc.fit_transform(x_test)


#Fit Logistic Regressor
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear", random_state = 0)
lr.fit(x_train, y_train)

#Predict for x_test
y_pred = lr.predict(x_test)

#Calculating accuracy by confusion matrix
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)

acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy score: ", acc)

pre = metrics.precision_score(y_test, y_pred)
print("Precission score: ", pre)

rec = metrics.recall_score(y_test, y_pred)
print("Recall score: ", rec)


#Creating a classification report
from sklearn.metrics import classification_report
cm = classification_report(y_test, y_pred)
print(cm)
   
