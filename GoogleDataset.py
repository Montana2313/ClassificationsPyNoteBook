#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:47:20 2020

@author: mac
"""

import pandas as pd
from matplotlib import pyplot as plt

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

training_df = pd.read_csv("california_housing_train.csv")

training_df["median_house_value"] /= 1000.0


totalRooms = training_df.iloc[:,4:6]
values = training_df.iloc[:,8:]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(totalRooms,values,test_size=0.33,random_state=1)

from sklearn.tree import DecisionTreeRegressor
dct_ = DecisionTreeRegressor(random_state = 1)
dct_.fit(x_train,y_train)
y_pred = dct_.predict(x_test)
'''
from sklearn.ensemble import RandomForestRegressor
rfc_object = RandomForestRegressor(n_estimators=10)
rfc_object.fit(x_train,y_train)
y_pred = rfc_object.predict(x_test)

from sklearn.linear_model import LinearRegression
lgobject = LinearRegression()
lgobject.fit(x_train,y_train)
y_pred = lgobject.predict(x_test)
'''
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))


import statsmodels.api as sm
model = sm.OLS(y_pred,y_test)
print(model.fit().summary())














