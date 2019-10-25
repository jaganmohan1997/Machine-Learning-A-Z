# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:10:49 2019

@author: Jagan Mohan
"""
from os import chdir, getcwd
chdir('C:/Users/user4/Documents/Jagan/Machine Learning A-Z/Part 2 - Regression/Section 4 - Simple Linear Regression/Simple_Linear_Regression')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the Dataset
df = pd.read_csv('Salary_Data.csv')
X = df.iloc[:,0].values
y = df.iloc[:,1].values

#Since we have only one column let us convert X into a 2D array
X = X.reshape(-1,1)

# Missing Values
df.isna().sum() #Hence no missing values are present

#No categorical values are present, hence no LabelEncoding is required



# we shall check our simple linear regression with and without Encoding

#Without Encoding the data

#Splitting the data into train and test model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 1234)


#Importing the linear Regression module
from sklearn.linear_model import LinearRegression

# Initiating the regressor
model = LinearRegression()

#Fitting the model
model.fit(X, y)

# Checking to see how it fitted our training data
y_pred = model.predict(X_train)
plt.plot(X_train, y_pred)
plt.plot(X_train, y_train, 'ro')
#plt.clf()

#Model R2 score
r_sq = model.score(X,y) # Rsquared of 0.957 without normalization

# Coefficient and respective intercept
model.coef_
model.intercept_

#Let's check how our predictions look like
y_pred = model.predict(X_test)

plt.plot(X_test, y_pred)
plt.plot(X_test, y_test, 'ro')

#Hence model without normalization is completed

#With Encoding the data
model_wn = LinearRegression(normalize = True)

model_wn.fit(X, y)

r_sqwn = model_wn.score(X,y) # Rsquared of 0.957 with normalization - Not much difference there :p

model_wn.coef_
model_wn.intercept_
