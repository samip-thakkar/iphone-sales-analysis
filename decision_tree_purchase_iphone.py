# -*- coding: utf-8 -*-
"""

@author: Samip
"""

#Importing dataset 
import pandas as pd
df = pd.read_csv('iphone_purchase_records.csv')
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#Convert Gender to number
from sklearn.preprocessing import LabelEncoder
lr = LabelEncoder()
x[:, 0] = lr.fit_transform(x[:, 0])

#Splitting data into training data and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.75, random_state = 0)


#Feature scaling
from sklearn.preprocessing import StandardScaler
x_sc = StandardScaler()
x_train = x_sc.fit_transform(x_train)
x_test = x_sc.transform(x_test)

#Fit the Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
dtc.fit(x_train, y_train)

#Predict for x_test
y_pred = dtc.predict(x_test)

#Checking accuracy of Naive Bayes
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred) 
print(cm)
acc = metrics.accuracy_score(y_test, y_pred) 
print("Accuracy score:",acc)
pre = metrics.precision_score(y_test, y_pred) 
print("Precision score:",pre)
rec = metrics.recall_score(y_test, y_pred) 
print("Recall score:",rec)
