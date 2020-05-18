#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 23:54:58 2020

@author: mac
"""

'''
k tane en yakın komşuya bakarız
en yakın olanın feature'nı alır
yerine bakar gelen veriye göre  lazy learning
dağıtılırekn bakarken eager örnek bölge çıkararak gelen veriyi sınıflandırır
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_Train,y_train)

y_pred = neigh.predict(X_Test)
print(y_pred)
print(y_test)


from sklearn.metrics import confusion_matrix
conf_matx = confusion_matrix(y_test,y_pred)
print(conf_matx)

print("SVM------------------------------")
'''
Sınıflandırmada ara margin noktaların da eleman kabul etmemesi Hard margin
Edip bunları hata olarak kabul etmesi durumu ise soft margin
'''
from sklearn.svm import SVC
svc_object = SVC(kernel= "sigmoid")
svc_object.fit(X_Train,y_train)
y_pred2 = svc_object.predict(X_Test)
print(y_pred2)
print(y_test)
conf_matrixSVM = confusion_matrix(y_test,y_pred2)
print(conf_matrixSVM)

print("SVM KERNEl TRİCK ---------------------")
'''
Kernel trick bildiğim üzere ayrılayamayacak şekilde  dağılmış olan veriler için kullanılan
başka bir boyuta çekerek ayrılmasını yeni sınırların belirlenmesi sağlanabilir
rbf örnek olucak olursa herhangi seçilen bir kernel noktasının yakınındakiler yukarı 
uzağındakiler aşağı olarak dağıtılması durumunda 3. boyutta bu veri ayrılabilir bir noktada olabilmektedir.
Bu sayede gelen veri hangi tarafta olmasına karşın ayrılabilmektedir.
'''













