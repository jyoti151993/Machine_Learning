# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 11:24:58 2023

@author: dbda-lab
"""

from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV , KFold
from sklearn.metrics import silhouette_score

import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")


#################### Milk data Set ###########################################

milk= pd.read_csv("milk.csv", index_col=0)
scaler= StandardScaler()
milkscaled=scaler.fit_transform(milk)


dbs=DBSCAN(eps=.8, min_samples=3)
dbs.fit(milkscaled)
print(dbs.labels_)  ## label= -1 is for outliers
len(set(dbs.labels_))

milk['Cluster']=dbs.labels_


##############################

epsilons=np.linspace(0.01,1,10)
min_points=[2,3,4,5]
sil=[]
for i in epsilons:
  for j in min_points:
      dbs=DBSCAN(eps=i,min_samples=j)
      dbs.fit(milkscaled)
      if len(set(dbs.labels_))>=3:
       score=silhouette_score(milkscaled, dbs.labels_)
       lst=[i,j,score]
       sil.append(lst)


pd_params=pd.DataFrame(sil, columns=['eps','min_pts','Sil_score'])

pd_params=pd_params.sort_values(by='Sil_score',ascending=False)

dbs=DBSCAN(eps=1,min_samples=3)
dbs.fit(milkscaled)
print(dbs.labels_)


silhouette_score(milkscaled, dbs.labels_)
milk["Cluster"]=dbs.labels_


####################### Nutrients ############

nutrient= pd.read_csv("nutrient.csv", index_col=0)
scaler= StandardScaler()
nutientscaled=scaler.fit_transform(nutrient)


epsilons=np.linspace(0.01,1,10)
min_points=[2,3,4,5]
sil=[]
for i in epsilons:
  for j in min_points:
      dbs=DBSCAN(eps=i,min_samples=j)
      dbs.fit(nutientscaled)
      if len(set(dbs.labels_))>=3:
       score=silhouette_score(nutientscaled, dbs.labels_)
       lst=[i,j,score]
       sil.append(lst)


pd_params=pd.DataFrame(sil, columns=['eps','min_pts','Sil_score'])

pd_params=pd_params.sort_values(by='Sil_score',ascending=False)

dbs=DBSCAN(eps=1,min_samples=2)
dbs.fit(nutientscaled)
print(dbs.labels_)


silhouette_score(nutientscaled, dbs.labels_)

nutrient["Cluster"]=dbs.labels_

############### USArrest ###################
usa= pd.read_csv("USArrests.csv", index_col=0)
scaler= StandardScaler()
usascaled=scaler.fit_transform(usa)


epsilons=np.linspace(0.01,1,10)
min_points=[2,3,4,5]
sil=[]
for i in epsilons:
  for j in min_points:
      dbs=DBSCAN(eps=i,min_samples=j)
      dbs.fit(usascaled)
      if len(set(dbs.labels_))>=3:
       score=silhouette_score(usascaled, dbs.labels_)
       lst=[i,j,score]
       sil.append(lst)


pd_params=pd.DataFrame(sil, columns=['eps','min_pts','Sil_score'])

pd_params=pd_params.sort_values(by='Sil_score',ascending=False)

dbs=DBSCAN(eps=1,min_samples=4)
dbs.fit(usascaled)
print(dbs.labels_)


silhouette_score(usascaled, dbs.labels_)
usa["Cluster"]=dbs.labels_

###################
import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Recency Frequency Monetary")

rfm= pd.read_csv("rfm_data_customer.csv", index_col=0)
rfm=rfm.drop("most_recent_visit",axis=1)

scaler= StandardScaler()
rfmscaled=scaler.fit_transform(rfm)


epsilons=np.linspace(0.01,1,10)
min_points=[2,3,4,5]
sil=[]
for i in epsilons:
  for j in min_points:
      dbs=DBSCAN(eps=i,min_samples=j)
      dbs.fit(rfmscaled)
      if len(set(dbs.labels_))>=3:
       score=silhouette_score(rfmscaled, dbs.labels_)
       lst=[i,j,score]
       sil.append(lst)


pd_params=pd.DataFrame(sil, columns=['eps','min_pts','Sil_score'])

pd_params=pd_params.sort_values(by='Sil_score',ascending=False)

dbs=DBSCAN(eps=1,min_samples=4)
dbs.fit(rfmscaled)
print(dbs.labels_)


silhouette_score(rfmscaled, dbs.labels_)
rfm["Cluster"]=dbs.labels_