# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 11:40:28 2023

@author: nikhilve
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Reading the CSV data/importing the data.
data = pd.read_csv('Mall_Customers.csv')


#Matrix of columns according to our data.
#Since it is Unsupervised learning we dont need y or Matrix of IV.
x=data.iloc[:,[3,4]].values

# Feature scaling is also not needed because mostlt the salary is betwnne 0-100 and the spending score is also btw 0-100.

#Creating the dendrogram to judge the ideal no of clusters/groups.
from scipy.cluster import hierarchy as sch
dendrogram= sch.dendrogram(sch.linkage(x, method='ward', metric='euclidean'))

plt.figure()
plt.subplot()#This plots the graph of wcss vs no of clusters. Here 1=no of rows, 2= no of columns and 1= 1st plot i.e. elbow graph

plt.title("Dendrogram for customer-grouping")
plt.xlabel('Customers')
plt.ylabel('Eucledian Distance')

# Implementing Hierarchical Clustering algorithm
from sklearn.cluster import AgglomerativeClustering as ac
hc = ac(n_clusters=5,affinity='euclidean', linkage='ward')

#applying this logic to our dataset, for actual grouping of customers.
y_hc = hc.fit_predict(x)

#Visual Representation
plt.subplot()
plt.scatter(x[y_hc==0,0], x[y_hc==0,1], s=80, c='red', label='HI-LS')
plt.scatter(x[y_hc==1,0], x[y_hc==1,1], s=80, c='green', label='AI-AS')
plt.scatter(x[y_hc==2,0], x[y_hc==2,1], s=80, c='blue', label='HI-HS')
plt.scatter(x[y_hc==3,0], x[y_hc==3,1], s=80, c='orange', label='LI-HS')
plt.scatter(x[y_hc==4,0], x[y_hc==4,1], s=80, c='pink', label='LS-LS')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Clustering')
plt.legend()



