# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

'''
--- Navie Bayes ---

Koşullu olasılık ile gerçekleşmektedir. Yağmur yağarken şemsiye alma ihitmalim gibi

Tüm olasıklar hesaplanır label değerlerine göre


gauess _ > Sürekli değerler için geçerli 
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

from sklearn.naive_bayes import GaussianNB
gauessianNavieBayes = GaussianNB()
gauessianNavieBayes.fit(x_train,y_train)
y_pred = gauessianNavieBayes.predict(x_test)

from sklearn.metrics import confusion_matrix
conf_matx = confusion_matrix(y_test,y_pred)
print(conf_matx)
right_Value = ((conf_matx[0][0] + conf_matx[1][1]) / 8) * 100
print("Accoury - >" + str(right_Value))

