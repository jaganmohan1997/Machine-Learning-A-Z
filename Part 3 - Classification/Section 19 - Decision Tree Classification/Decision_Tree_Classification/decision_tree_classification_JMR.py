# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:09:54 2019

@author: user4
"""



# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the Dataset
df = pd.read_csv('Social_Network_Ads.csv')
X = df.iloc[:,2:4].values
y = df.iloc[:,4].values

# Missing Values
df.isna().sum() #we have no issing values, hence no imputation required


#from sklearn.impute import SimpleImputer
#imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
#X.iloc[:,1:3] = imputer.fit_transform(X.iloc[:,1:3])


# Categorical Data
# We have one categorical data that is Gender. let us label it and one hot encode it
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_x = LabelEncoder()
X[:,0] = label_encoder_x.fit_transform(X[:,0])
# As it has only two labels one hot encoding is not required
#one_hot_encoder = OneHotEncoder(categorical_features = [0])
#X = one_hot_encoder.fit_transform(X).toarray()
#label_encoder_y = LabelEncoder()
#y = label_encoder_y.fit_transform(y)


# Train and Test Split the Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, random_state = 1234)


# Feature Scaling them all
from sklearn.preprocessing import StandardScaler
standard_scaler_x = StandardScaler()
X_train = standard_scaler_x.fit_transform(X_train)
X_test = standard_scaler_x.transform(X_test)




# Building the classifier
from sklearn.tree import DecisionTreeClassifier

#Using euclidean distance in the algorithm
classifier = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, min_samples_split=10)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


# Let's use a confusion matrix to see the accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
accuracy
#Data is just 400 and we can tune more if only we have more observations and features.

















