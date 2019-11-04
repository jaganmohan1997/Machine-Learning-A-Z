# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:31:23 2019

@author: user4
"""


# Importing the Libraries
import numpy as np
import pandas as pd

# Importing the dataset
df = pd.read_csv('Position_Salaries.csv')

X = df.iloc[:,1:2].values
y = df.iloc[:,2].values

#Fitting the Decision Tree Regressor model
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 15)

regressor.fit(X,y)
y_pred = regressor.predict(X)


#Lets do a quick graph
import matplotlib.pyplot as plt
plt.plot(X,y,'bo')
plt.plot(X,y_pred)

y_custom = regressor.predict([[6.5]])
X_test = np.arange(0,10,0.05).reshape((-1,1))

y_pred_test = regressor.predict(X_test)
plt.plot(X,y,'bo')
plt.plot(X_test,y_pred_test)

#Well the data is very small to do a Random Forest regressor on it
# Above is just an example of how to code a Random Forest Regressor
