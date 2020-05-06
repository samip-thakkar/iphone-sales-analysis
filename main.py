# -*- coding: utf-8 -*-
"""

@author: Samip
"""

from preprocess import PreProcess
from classifier import Classifier
from modelEvaluation import ModelEvaluation

#Create the objects
pp = PreProcess()
classifier = Classifier()
me = ModelEvaluation()

#Preprocess the data
x_train, x_test, y_train, y_test = pp.scale_data()

choice = int(input("Enter 1 for Logistic Regression, 2 for Decision Tree Classifier, 3 for KNN, 4 for Naive Bayes, 5 for Random Forest, 6 for SVM, 7 for XG Boost, 8 for Adaptive Boosting, 9 for LDA: "))
clf = {1: classifier.logistic_regression, 2: classifier.decision_tree_classifer, 3: classifier.knn, 4: classifier.naive_bayes, 5: classifier.random_forest, 6: classifier.svm, 7: classifier.xg_boost, 8: classifier.ada_boost, 9: classifier.lda}
model = clf[choice](x_train, y_train)

#Get the predicted values
y_pred = model.predict(x_test)

#Get the model evaluation
me.modelevaluation(y_test, y_pred)

#Get the ROC Curve
me.plotROC(y_test, y_pred)

gender = int(input("Enter your gender, 1 for male and 0 for female: "))
age = int(input("Enter the age: "))
salary = int(input("Enter your salary: "))

pred = model.predict([[gender, age, salary]])
if pred == 1:
    print("You will buy i-phone")
elif pred == 0:
    print("You won't buy right now")