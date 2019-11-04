# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:49:48 2019

@author: user4
"""

# Importing the Libraries
import numpy as np
import pandas as pd

# Importing the dataset
df = pd.read_csv('Position_Salaries.csv')# We see that we have a non linear regression problem hopefully polynomial kernal will do for us

X = df.iloc[:,1:2].values
y = df.iloc[:,2].values

#Fitting the SVR model
from sklearn.svm import SVR

regressor = SVR(kernel = 'poly', degree = 3, epsilon = 0.1, C = 1)
regressor.fit(X,y)
y_pred = regressor.predict(X)
#Lets do a quick graph
import matplotlib.pyplot as plt
plt.plot(X,y,'bo')
plt.plot(X,y_pred)
# Lets try further optmization parameters



regressor = SVR(kernel = 'poly', degree = 3, epsilon = 0.1, C = 10)
regressor.fit(X,y)
y_pred = regressor.predict(X)
#Lets do a quick graph
import matplotlib.pyplot as plt
plt.plot(X,y,'bo')
plt.plot(X,y_pred)
# Lets try further optmization parameters


regressor = SVR(kernel = 'poly', degree = 3, epsilon = 0.1, C = 100)
regressor.fit(X,y)
y_pred = regressor.predict(X)
#Lets do a quick graph
import matplotlib.pyplot as plt
plt.plot(X,y,'bo')
plt.plot(X,y_pred)
# Lets try further optmization parameters. We don't see much difference in increase from C 10 -100. Lets change the degree with C 10




regressor = SVR(kernel = 'poly', degree = 4, epsilon = 1, C = 10)
regressor.fit(X,y)
y_pred = regressor.predict(X)
#Lets do a quick graph
import matplotlib.pyplot as plt
plt.plot(X,y,'bo')
plt.plot(X,y_pred)
# Lets try further optmization parameters


regressor = SVR(kernel = 'poly', degree = 5, epsilon = 1, C = 10)
regressor.fit(X,y)
y_pred = regressor.predict(X)
#Lets do a quick graph
import matplotlib.pyplot as plt
plt.plot(X,y,'bo')
plt.plot(X,y_pred)
# Lets try further optmization parameters


regressor = SVR(kernel = 'poly', degree = 6, epsilon = 1, C = 10)
regressor.fit(X,y)
y_pred = regressor.predict(X)
#Lets do a quick graph
import matplotlib.pyplot as plt
plt.plot(X,y,'bo')
plt.plot(X,y_pred)
# Lets try further optmization parameters
#This is a very good optimization compared to from where we start, but degree 6 is computationally heavy
#Let's increase the epsilon to reduce the variance and we will call it a day



regressor = SVR(kernel = 'poly', degree = 5, epsilon = 10, C = 10)
regressor.fit(X,y)
y_pred = regressor.predict(X)
#Lets do a quick graph
import matplotlib.pyplot as plt
plt.plot(X,y,'bo')
plt.plot(X,y_pred)
# Perfect. That looks good optimization of parameters
