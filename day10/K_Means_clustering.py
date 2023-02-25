# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 19:06:36 2023

@author: dbda-lab
"""

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")

milk= pd.read_csv("milk.csv", index_col=0)
scaler= StandardScaler()
milkscaled=scaler.fit_transform(milk)
km= KMeans(n_clusters=5, random_state=2023)
km.fit(milkscaled)
labels=km.predict(milkscaled)
print(labels)

milk['Cluster']=labels


##################### with out loop go on changing the n_clusters till 24 as we have total 25 obs, find the inertia
## lesser the inertia better is the cluster
milk= pd.read_csv("milk.csv", index_col=0)
scaler= StandardScaler()
milkscaled=scaler.fit_transform(milk)
km= KMeans(n_clusters=2, random_state=2023)
km.fit(milkscaled)
print(km.inertia_)

km= KMeans(n_clusters=2, random_state=2023)
km.fit(milkscaled)
print(km.inertia_)



############# With Loop ################
wss=[]
for i in np.arange(2,11):
    km= KMeans(n_clusters=i, random_state=2023)
    km.fit(milkscaled)
    print(km.inertia_)
    wss.append(km.inertia_)
    
plt.plot(np.arange(2,11),wss)
plt.scatter(np.arange(2,11),wss)
plt.xlabel("Number of Clusters")
plt.ylabel("Within Sums of Squares")

plt.show()

km= KMeans(n_clusters=4, random_state=2023)
km.fit(milkscaled)
labels=km.predict(milkscaled)
print(labels)

milk['Cluster']=labels
milk=milk.sort_values(by='Cluster')

milk.groupby('Cluster').mean()


################

nutrient= pd.read_csv("nutrient.csv", index_col=0)
scaler= StandardScaler()
nutientscaled=scaler.fit_transform(nutrient)

wss=[]
for i in np.arange(2,11):
    km= KMeans(n_clusters=i, random_state=2023)
    km.fit(nutientscaled)
    print(km.inertia_)
    wss.append(km.inertia_)
    
plt.plot(np.arange(2,11),wss)
plt.scatter(np.arange(2,11),wss)
plt.xlabel("Number of Clusters")
plt.ylabel("Within Sums of Squares")

plt.show()



km= KMeans(n_clusters=5, random_state=2023)
km.fit(nutientscaled)
labels=km.predict(nutientscaled)
print(labels)

nutrient['Cluster']=labels
nutrient=nutrient.sort_values(by='Cluster')

nutrient.groupby('Cluster').mean()
