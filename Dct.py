#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:06:01 2020

@author: mac
"""

'''
--- Karar ağaçları algoritması Sınıflandırıcı

ID3 algoritması entropy üzerinden gain hesabı yaparak en uygun yolu bulunur.

Bildiğim kısaca
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calcConf(matrix , how_many):
    result = ((matrix[0][0] + matrix[1][1]) / how_many) * 100
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


veriler = pd.read_csv('veriler.csv')
X_ =  veriler.iloc[:,1:4].values # bağımsızlar
Y_ =  veriler.iloc[:,4:].values # bağımlılar

from sklearn.tree import DecisionTreeClassifier
# creation gini veya entropy olarak seçilebiliyor.
# gini entropy'den farklı olarak - pi^2 olarak hesaplanmaktadırç. log2 yerine


dct = DecisionTreeClassifier(random_state = 0 , criterion = 'entropy')

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X_,Y_,test_size=0.33,random_state=1)

dct.fit(x_train,y_train)
y_pred = dct.predict(x_test)

from sklearn.metrics import confusion_matrix
conf = confusion_matrix(y_test,y_pred)
print(conf)

calcConf(conf,8)
















