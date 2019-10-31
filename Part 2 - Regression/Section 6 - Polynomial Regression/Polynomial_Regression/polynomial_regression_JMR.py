# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 16:55:59 2019

@author: user4
"""

# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
df = pd.read_csv('Position_Salaries.csv')

X = df.iloc[:,1:2].values
y = df.iloc[:,2].values


# Creating a linear regression and fitting our data perfectly
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X,y)

y_pred = regressor.predict(X)

plt.plot(X,y_pred,'r')
plt.plot(X,y,'bo')

#Ouch, That didn't go at all as we expected, let's try polynomial features for gods sake 


# Creating polynomial features for the dataset
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 4)
X_poly = poly.fit_transform(X)

# Creating a linear regression on polynomial features i.e. polynomial linear regression :p
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_poly,y)

y_pred_poly = regressor.predict(X_poly)

plt.plot(X,y_pred_poly,'r')
plt.plot(X,y,'bo')

#Welll, I hope that's a good way to use a polynomial Linear Regression 