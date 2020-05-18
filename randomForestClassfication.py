#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:14:25 2020

@author: mac
"""

'''
Random Forest
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calcConf(matrix , how_many):
    result = ((matrix[0][0] + matrix[1][1]) / how_many) * 100
    print("Accoury - >" + str(result))

veriler = pd.read_csv('veriler.csv')
X_ =  veriler.iloc[:,1:4].values # bağımsızlar
Y_ =  veriler.iloc[:,4:].values # bağımlılar

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X_,Y_,test_size=0.33,random_state=1)

from sklearn.preprocessing import StandardScaler
# normalizasyon olayı aslında 
standardScaler = StandardScaler()
X_Train = standardScaler.fit_transform(x_train)
X_Test = standardScaler.transform(x_test) 
# fit olmamasını sebebi yukarıda öğrendiğini uygula demek


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 10,criterion = "entropy")
rfc.fit(x_train,y_train)

y_pred = rfc.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
calcConf(cm , 8)








