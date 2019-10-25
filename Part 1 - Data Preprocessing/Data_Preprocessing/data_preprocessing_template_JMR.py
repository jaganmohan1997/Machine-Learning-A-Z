# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:09:56 2019

@author: Jagan Mohan
"""

# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the Dataset
df = pd.read_csv('Data.csv')
X = df.iloc[:,:3]
y = df.iloc[:,3]

# Missing Values
df.isna().sum() #we have to missing values each one in Age & Salary Columns

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
X.iloc[:,1:3] = imputer.fit_transform(X.iloc[:,1:3])


# Categorical Data
# We have one categorical data that is country let us label it and one hot encode it
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_x = LabelEncoder()
X.iloc[:,0] = label_encoder_x.fit_transform(X.iloc[:,0])
one_hot_encoder = OneHotEncoder(categorical_features = [0])
X = one_hot_encoder.fit_transform(X).toarray()
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)


# Train and Test Aplit the Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, random_state = 1234)


# Feature Scaling them all
from sklearn.preprocessing import StandardScaler
standard_scaler_x = StandardScaler()
X_train = standard_scaler_x.fit_transform(X_train)
X_test = standard_scaler_x.transform(X_test)


#Hence concludes the sample of Preprocessing Template in Python