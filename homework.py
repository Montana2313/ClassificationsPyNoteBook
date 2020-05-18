# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def calcConf(matrix , how_many):
    result = ((matrix[0][0] + matrix[1][1] + matrix[2][2]) / how_many) * 100
    print("Accoury - >" + str(result))
    TN = matrix[0][0] 
    FP = matrix[0][1]
    FN = matrix[1][0]
    TP = matrix[1][1]
    sensivty = TP / (TP + FN)
    specificty = TN / (FP + TN)
    precision = TP / (TP + FP)
    print("sensivity -> " + str(sensivty))
    print("specifity -> " + str(specificty))
    print("precision -> " + str(precision))



veriler = pd.read_excel('Iris.xls')

print(veriler.head(5))

bagimsiz_degiskenler = veriler.iloc[:,:4]
bagimli_degisken = veriler.iloc[:,4:]

print(bagimsiz_degiskenler.describe())
print(bagimli_degisken.describe())



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(bagimsiz_degiskenler,bagimli_degisken,test_size=0.33,random_state=1)


print("Logistic Regression -----------------")
from sklearn.linear_model import LogisticRegression
lg_object = LogisticRegression(random_state = 0)
lg_object.fit(x_train,y_train)

y_pred = lg_object.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
calcConf(cm,50)

print("Decision Tree Classifier -----------------")
from sklearn.tree import DecisionTreeClassifier
dct = DecisionTreeClassifier(criterion = "entropy")
dct.fit(x_train,y_train)
y_predDCT = dct.predict(x_test)
cm_DCT = confusion_matrix(y_test,y_predDCT)
print(cm_DCT)
calcConf(cm_DCT,50)


print("Random Forest Classifier -----------------")
from sklearn.ensemble import RandomForestClassifier
rdt = RandomForestClassifier(criterion = "entropy",n_estimators = 5)
rdt.fit(x_train,y_train)
y_pred_RDT = rdt.predict(x_test)
cm_RDT = confusion_matrix(y_test,y_pred_RDT)
print(cm_RDT)
calcConf(cm_RDT,50)

print("KNN -----------------")
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 4)
knn.fit(x_train,y_train)
y_pred_KNN = knn.predict(x_test)
cm_KNN = confusion_matrix(y_test,y_pred_KNN)
print(cm_KNN)
calcConf(cm_KNN,50)

print("GaussionNB -----------------")

from sklearn.naive_bayes import GaussianNB
gauessianNavieBayes = GaussianNB()
gauessianNavieBayes.fit(x_train,y_train)
y_pred_NAVIEBAYES = gauessianNavieBayes.predict(x_test)
cm_NAVIEBAYES = confusion_matrix(y_test,y_pred_NAVIEBAYES)
print(cm_NAVIEBAYES)
calcConf(cm_NAVIEBAYES,50)



















