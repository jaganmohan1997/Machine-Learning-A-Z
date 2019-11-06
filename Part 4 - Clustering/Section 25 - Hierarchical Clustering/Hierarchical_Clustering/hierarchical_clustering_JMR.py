# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:05:29 2019

@author: user4
"""


#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the datasets
df = pd.read_csv('Mall_Customers.csv')
X = df.iloc[:,3:5]
#X = X.astype(float)

#Missing Values
df.isna().sum() # We don't have any missing values

#using a dendogram to find the optimal number of clusters
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
Z = linkage(X, 'ward')
Z = Z.astype('double')
dg = dendrogram(Z)
plt.show()
# Hence by long vertical without horizontal cuts method, the number of clusters is 5

#Lets buid agglomerative cluster using 5 cluters
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage= 'ward')
y_hc = hc.fit_predict(X)

X = X.values
X = X.astype('float')
#Vizualizing our clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 25, c = 'red')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 25, c = 'blue')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 25, c = 'green')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 25, c = 'cyan')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 25, c = 'black')
plt.show()
#hence we conclude the hierarchical Clustering