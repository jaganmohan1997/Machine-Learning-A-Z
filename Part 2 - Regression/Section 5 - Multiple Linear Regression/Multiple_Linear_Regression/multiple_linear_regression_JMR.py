# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:28:33 2019

@author: Jagan Mohan
"""

#Importing the required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
df = pd.read_csv('50_Startups.csv')
df.iloc[:,3].value_counts()

X = df.iloc[:,:4].values
y = df.iloc[:,-1:].values

#Checking for any Missing Values
df.isnull().sum() # Since there are no missing values no imputation is required

#Lets encode the categorical variable that is State
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
X[:,3] = labelencoder_x.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features= [3])
X = onehotencoder.fit_transform(X).toarray()[:,1:]#Escaping the dummy variable trap

#Now that the required data is prepared let's split the training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 1234)

# Let's fit the regression model on our Train Data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting our Model
y_pred = regressor.predict(X_test)

y_pred_train = regressor.predict(X_train)

plt.plot(y_test, y_pred, 'ro')

plt.plot(y_train, y_pred_train, 'bo')

#Building an optimal model using Backward Propagation
import statsmodels.api as sm

X = np.append(arr = np.ones((50,1), dtype = 'int64'), values = X, axis = 1)

X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(y,X_opt).fit()
regressor_OLS.summary()
#We see that variable x2 has high p -value (>0.05 sls) and hence has to be removed

X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(y,X_opt).fit()
regressor_OLS.summary()
#We see that variable x1 has high p -value (>0.05 sls) and hence has to be removed

X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(y,X_opt).fit()
regressor_OLS.summary()
#We see that variable x2 has high p -value (>0.05 sls) and hence has to be removed

X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(y,X_opt).fit()
regressor_OLS.summary()
#We see that variable x2 has high p -value (>0.05 sls) and hence has to be removed

X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(y,X_opt).fit()
regressor_OLS.summary()
#Hence only the amount that we spend in research is useful for us




































