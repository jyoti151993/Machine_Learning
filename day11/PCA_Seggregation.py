# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 16:49:17 2023

@author: dbda-lab
"""

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.decomposition import PCA
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

km=KMeans(n_clusters=best_k, random_state=2023)
labels=km.fit_predict(milkscaled)
print(labels)

prcomp=PCA()
components=prcomp.fit_transform(milkscaled)
pd_comp=pd.DataFrame(components, columns=['PC1',"PC2",'PC3','PC4','PC5'])

pd_comp['Cluster']=labels
pd_comp['Cluster']=pd_comp['Cluster'].astype(object)
pc1=pd_comp['PC1'].values  # numpy format
pc2=pd_comp['PC2'].values
sns.scatterplot(data=pd_comp, x='PC1', y='PC2', palette='Dark2', hue='Cluster')
for i , text in enumerate(list(milk.index)):
    plt.annotate(text, (pc1[i],pc2[i]), fontsize=8)
plt.show()


########## Nutrients 
import pandas as pd
nutrient= pd.read_csv("nutrient.csv", index_col=0)
scaler= StandardScaler()
nutrientscaled=scaler.fit_transform(nutrient)

sil=[]
for i in np.arange(2,11):
    km=KMeans(n_clusters=i,random_state=2023)
    km.fit(nutrientscaled)
    labels=km.predict(nutrientscaled)
    score=silhouette_score(nutrientscaled, labels)
    sil.append(score)
    
    
i_max=np.argmax(sil) 
ks=np.arange(2,11)
best_k=ks[i_max]
print("Best K=",best_k)
print("Best Score=", np.max(sil))

plt.plot(np.arange(2,11),sil)

km=KMeans(n_clusters=best_k, random_state=2023)
labels=km.fit_predict(nutrientscaled)
print(labels)

prcomp=PCA()
components=prcomp.fit_transform(nutrientscaled)
pd_comp=pd.DataFrame(components, columns=['PC1',"PC2",'PC3','PC4','PC5'])

pd_comp['Cluster']=labels
pd_comp['Cluster']=pd_comp['Cluster'].astype(object)
pc1=pd_comp['PC1'].values  # numpy format
pc2=pd_comp['PC2'].values
sns.scatterplot(data=pd_comp, x='PC1', y='PC2', palette='Dark2', hue='Cluster')
for i , text in enumerate(list(nutrient.index)):
    plt.annotate(text, (pc1[i],pc2[i]), fontsize=8)
plt.show()



################ Img _Segmention ###############
image_seg=pd.read_csv("Image_Segmention.csv")

x=image_seg.drop('Class',axis=1)
y=image_seg['Class']
x_scaled=scaler.fit_transform(x)

prcomp=PCA()
components=prcomp.fit_transform(x_scaled)
pd_comp=pd.DataFrame(components[:,:2],columns=['PC1','PC2'])
print(np.cumsum(prcomp.explained_variance_ratio_*100))
pd_comp['Class']=y

sns.scatterplot(data=pd_comp,x='PC1',palette='Dark2',y='PC2',hue='Class')
plt.show()

############# Wnconsin ##################
cancer=pd.read_csv("BreastCancer.csv")
x=cancer.drop(["Class","Code"],axis=1)
y=cancer['Class']

x_scaled=scaler.fit_transform(x)

prcomp=PCA()
components=prcomp.fit_transform(x_scaled)
pd_comp=pd.DataFrame(components[:,:2],columns=['PC1','PC2'])
print(np.cumsum(prcomp.explained_variance_ratio_*100))
pd_comp['Class']=y

sns.scatterplot(data=pd_comp,x='PC1',palette='Dark2',y='PC2',hue='Class')
plt.show()
