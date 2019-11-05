# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:38:30 2019

@author: user4
"""

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the datasets
df = pd.read_csv('Mall_Customers.csv')
X = df.iloc[:,3:5].values
X = X.astype(float)

#Missing Values
df.isna().sum() # We don't have any missing values

#Since we are trying to find clusters, and we have all coloumns as continuous let's scale them 
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X[:,0:3] = ss.fit_transform(X[:,0:3])

#We don't have any particular Training and Test datasets as we are trying to establish a cluster

#Building the Cluster
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, init = 'k-means++')
kmeans.fit(X)
kmeans.inertia_
#Now we can't know if it is the best model as we need to compare with other models

def kmeans_optimizer(n):
    wcss = []
    for i in np.arange(1,n+1,1):
        kmeans = KMeans(n_clusters=i, init = 'k-means++')
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)   
    plt.plot(range(1,n+1), wcss)

kmeans_optimizer(10)
#so k = 5 seems to be a good number
        
kmeans = KMeans(n_clusters=5, init = 'k-means++')
kmeans.fit(X)

#Let's plot and visualize
y_means = kmeans.fit_predict(X)

plt.scatter(X[y_means == 0,0], X[y_means == 0,1], s = 50, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_means == 1,0], X[y_means == 1,1], s = 50, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_means == 2,0], X[y_means == 2,1], s = 50, c = 'orange', label = 'Cluster 3')
plt.scatter(X[y_means == 3,0], X[y_means == 3,1], s = 50, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_means == 4,0], X[y_means == 4,1], s = 50, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 150, c = 'green', label = 'Centers')
plt.title('Cluster Visualization')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
#This concludes our Clustering for the day. We can see that different groups of people and interpret their behaviour from the plot