# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 09:26:57 2023

@author: dbda-lab
"""

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")


#################### Milk data Set ###########################################

milk= pd.read_csv("milk.csv", index_col=0)
scaler= StandardScaler()
milkscaled=scaler.fit_transform(milk)

sil=[]
for i in np.arange(2,11):
    km=KMeans(n_clusters=i,random_state=2023)
    km.fit(milkscaled)
    labels=km.predict(milkscaled)
    score=silhouette_score(milkscaled, labels)
    sil.append(score)
    
    
i_max=np.argmax(sil) 
ks=np.arange(2,11)
best_k=ks[i_max]
print("Best K=",best_k)
print("Best Score=", np.max(sil))

plt.plot(np.arange(2,11),sil)


################## nutients ########################################

nutrient= pd.read_csv("nutrient.csv", index_col=0)
scaler= StandardScaler()
nutientscaled=scaler.fit_transform(nutrient)

sil=[]
for i in np.arange(2,11):
    km=KMeans(n_clusters=i,random_state=2023)
    km.fit(nutientscaled)
    labels=km.predict(nutientscaled)
    score=silhouette_score(nutientscaled, labels)
    sil.append(score)
    
    
i_max=np.argmax(sil) 
ks=np.arange(2,11)
best_k=ks[i_max]
print("Best K=",best_k) # 4
print("Best Score=", np.max(sil))  # 0.418

plt.plot(np.arange(2,11),sil)



############### USArrest #########################################
usa= pd.read_csv("USArrests.csv", index_col=0)
scaler= StandardScaler()
usa_scaled=scaler.fit_transform(usa)

sil=[]
for i in np.arange(2,11):
    km=KMeans(n_clusters=i,random_state=2023)
    km.fit(usa_scaled)
    labels=km.predict(usa_scaled)
    score=silhouette_score(usa_scaled, labels)
    sil.append(score)
    
    
i_max=np.argmax(sil) 
ks=np.arange(2,11)
best_k=ks[i_max]
print("Best K=",best_k) # k=2
print("Best Score =", np.max(sil))  # 0.4084890326217641

plt.plot(np.arange(2,11),sil)

###########################   Recency, Frequency & Monetory  Case study  using Silhouette score ##########################
import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Recency Frequency Monetary")

rfm= pd.read_csv("rfm_data_customer.csv", index_col=0)
rfm=rfm.drop("most_recent_visit",axis=1)

scaler= StandardScaler()
rfmscaled=scaler.fit_transform(rfm)

sil=[]
for i in np.arange(2,11):
    km=KMeans(n_clusters=i,random_state=2023)
    km.fit(rfmscaled)
    labels=km.predict(rfmscaled)
    score=silhouette_score(rfmscaled, labels)
    sil.append(score)
    
    
i_max=np.argmax(sil) 
ks=np.arange(2,11)
best_k=ks[i_max]
print("Best K=",best_k) # k=3
print("Best Score =", np.max(sil))  # 0.37088155788721333

plt.plot(np.arange(2,11),sil)


######################## Centroid /inertia K MEans  RFM Data set  ##############
rfm= pd.read_csv("rfm_data_customer.csv", index_col=0)
rfm=rfm.drop("most_recent_visit",axis=1)

scaler= StandardScaler()
rfmscaled=scaler.fit_transform(rfm)


wss=[]
for i in np.arange(2,11):
    km= KMeans(n_clusters=i, random_state=2023)
    km.fit(rfmscaled)
    print(km.inertia_)
    wss.append(km.inertia_)
    
plt.plot(np.arange(2,11),wss)
plt.scatter(np.arange(2,11),wss)
plt.xlabel("Number of Clusters")
plt.ylabel("Within Sums of Squares")

plt.show()


km= KMeans(n_clusters=3, random_state=2023)
km.fit(rfmscaled)
labels=km.predict(rfmscaled)
print(labels)

rfm['Cluster']=labels
rfm=rfm.sort_values(by='Cluster')

rfm.groupby('Cluster').mean()

################# Hierarchical Clustering Denogram using centroid Linkage  RFM Dataset  #
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler

import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Recency Frequency Monetary")

rfm= pd.read_csv("rfm_data_customer.csv", index_col=0)
rfm=rfm.drop("most_recent_visit",axis=1)

scaler= StandardScaler()
rfmscaled=scaler.fit_transform(rfm)


#calculate the linkage: merging
mergings = linkage(rfmscaled, method='centroid')
plt.figure(figsize=(12,8))


dendrogram(mergings, labels=np.array(rfm.index),leaf_rotation=45, leaf_font_size=10)
plt.show()

###################### Mini Batch K Means Clustering for large data set >>> fast computation #########
import os
from sklearn.cluster import MiniBatchKMeans
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Recency Frequency Monetary")

rfm= pd.read_csv("rfm_data_customer.csv", index_col=0)
rfm=rfm.drop("most_recent_visit",axis=1)

scaler= StandardScaler()
rfmscaled=scaler.fit_transform(rfm)

sil=[]
for i in np.arange(2,11):
    km=MiniBatchKMeans(n_clusters=i,random_state=2023)
    km.fit(rfmscaled)
    labels=km.predict(rfmscaled)
    score=silhouette_score(rfmscaled, labels)
    sil.append(score)
    
    
i_max=np.argmax(sil) 
ks=np.arange(2,11)
best_k=ks[i_max]
print("Best K=",best_k) # k=3
print("Best Score =", np.max(sil))  # 0.37088155788721333

plt.plot(np.arange(2,11),sil)

#################################################