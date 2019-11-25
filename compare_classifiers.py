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
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 0)


#Feature scaling
from sklearn.preprocessing import StandardScaler
x_sc = StandardScaler()
x_train = x_sc.fit_transform(x_train)
x_test = x_sc.transform(x_test)

#Compare all calssifiers
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

classification_models = []
classification_models.append(('Logistic Regression', LogisticRegression(solver="liblinear")))
classification_models.append(('K Nearest Neighbor', KNeighborsClassifier(n_neighbors=5, metric="minkowski",p=2)))
classification_models.append(('Kernel SVM', SVC(kernel = 'rbf',gamma='scale')))
classification_models.append(('Naive Bayes', GaussianNB()))
classification_models.append(('Decision Tree', DecisionTreeClassifier(criterion = "entropy")))
classification_models.append(('Random Forest', RandomForestClassifier(n_estimators=300, criterion="entropy")))

for name, model in classification_models:
    kfold = KFold(n_splits=10, random_state=3)
    result = cross_val_score(model, x, y, cv=kfold, scoring='accuracy')
    print("%s: Mean Accuracy = %.2f%% - SD Accuracy = %.2f%%" % (name, result.mean()*100, result.std()*100))
