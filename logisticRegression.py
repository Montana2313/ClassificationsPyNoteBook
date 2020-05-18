#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 22:00:55 2020

@author: mac
"""

'''
Logistic Regression : 
    Logistic regresyen bir sınıflandırma algoritmasıdır. Bu algoritma ile 
    etiketleri veya sayılar daha önce öğrendiğimiz gibi bir regresyon mantığı ile 
    hataları minimize etmeye çalışır. Lojistik bir fonksiyon çizer
    
    Formülü ; 
    1 / 1 + e^t
    t = ax + b 
    
    t bir multiple lineer regression gibi çoğaltılabilmektedir.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veriler = pd.read_csv('veriler.csv')
X_ =  veriler.iloc[:,1:4].values # bağımsızlar
Y_ =  veriler.iloc[:,4:].values # bağımlılar


print("-------- TEST VE TRAIN ---------")

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X_,Y_,test_size=0.33,random_state=1)

print("--------- ÖZNİTELİK ÖLÇEKLEME ------------")
'''
from sklearn.preprocessing import StandardScaler
# normalizasyon olayı aslında 
standardScaler = StandardScaler()
X_Train = standardScaler.fit_transform(x_train)
X_Test = standardScaler.transform(x_test) 
# fit olmamasını sebebi yukarıda öğrendiğini uygula demek
'''
from sklearn.linear_model import LogisticRegression
lg_object = LogisticRegression(random_state = 0)
lg_object.fit(x_train,y_train)

y_pred = lg_object.predict(x_test)
print(y_pred)
print(y_test)


from sklearn.metrics import confusion_matrix
conf_matx = confusion_matrix(y_test,y_pred)
print(conf_matx)
right_Value = ((conf_matx[0][0] + conf_matx[1][1]) / 8) * 100
print("Accoury - >" + str(right_Value))


















