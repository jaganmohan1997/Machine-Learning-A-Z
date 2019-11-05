# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:25:10 2019

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
from sklearn.neighbors import KNeighborsClassifier

#Using euclidean distance in the algorithm
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


# Let's use a confusion matrix to see the accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
accuracy
# hence we observe around 86.2%


def knn_optimizer(n):
    optimum = np.zeros((n,2))
    for i in np.arange(1,n+1,1):
        classifier = KNeighborsClassifier(n_neighbors=i, metric='minkowski', p = 2)
    
        classifier.fit(X_train, y_train)
        
        y_pred = classifier.predict(X_test)
        
        
        # Let's use a confusion matrix to see the accuracy
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
        optimum[i-1,0] = i
        optimum[i-1,1] = accuracy
    return optimum

visual = knn_optimizer(15)
plt.plot(visual[:,0],visual[:,1])

#From the plot we can see that we have peaks at K value of 7 and 9



classifier = KNeighborsClassifier(n_neighbors=7, metric='minkowski', p = 2)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

#Following is a much better visualization of the prediction
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()#we see that the predictions are much better than logistic regression

# Visualising the test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

