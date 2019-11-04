# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:18:30 2019

@author: user4
"""


# Importing the Libraries
import numpy as np
import pandas as pd

# Importing the dataset
df = pd.read_csv('Position_Salaries.csv')# We see that we have a non linear regression problem hopefully polynomial kernal will do for us

X = df.iloc[:,1:2].values
y = df.iloc[:,2].values

#Fitting the Decision Tree Regressor model
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor( random_state=0 )

regressor.fit(X,y)
y_pred = regressor.predict(X)


#Lets do a quick graph
import matplotlib.pyplot as plt
plt.plot(X,y,'bo')
plt.plot(X,y_pred)
# Lets try further optmization parameters

y_custom = regressor.predict([[6.5]])
X_test = np.arange(0,10,0.05).reshape((-1,1))

y_pred_test = regressor.predict(X_test)
plt.plot(X,y,'bo')
plt.plot(X_test,y_pred_test)

#Wohoo we just overfitted our model to each and every dataset. 
# Since we have very less observations, it will overfit..... Hence CART is better used for datasets with huge number of observations for better results
