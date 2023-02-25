# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 14:42:05 2023

@author: dbda-lab
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import pca
from pca import pca
import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")


#################### Milk data Set ###########################################

milk= pd.read_csv("milk.csv", index_col=0)
scaler= StandardScaler()
milkscaled=scaler.fit_transform(milk)

prcomp=PCA()
components=prcomp.fit_transform(milkscaled)

######## scaling is not a part of PCA algo its for better performance/result

pd_comp=pd.DataFrame(components, columns=['PC1',"PC2",'PC3','PC4','PC5'])

#pd_comp.var()

print(prcomp.explained_variance_)
print(prcomp.explained_variance_ratio_*100)
print(np.cumsum(prcomp.explained_variance_ratio_*100))

# Eigen vectors
print(prcomp.components_)

############## Biplot #############
milkscaled =pd.DataFrame(milkscaled, columns=milk.columns,index=milk.index)

model=pca()
results=model.fit_transform(milkscaled)
model.biplot(label=True, legend=False)


################
usa= pd.read_csv("USArrests.csv", index_col=0)
scaler= StandardScaler()
usascaled=scaler.fit_transform(usa)


prcomp=PCA()
components=prcomp.fit_transform(usascaled)

pd_comp=pd.DataFrame(components, columns=['PC1',"PC2",'PC3','PC4'])

#pd_comp.var()

print(prcomp.explained_variance_)
print(prcomp.explained_variance_ratio_*100)
print(np.cumsum(prcomp.explained_variance_ratio_*100))

# Eigen vectors
print(prcomp.components_)

usascaled=pd.DataFrame(usascaled, columns=usa.columns,index=usa.index)

model=pca()
results=model.fit_transform(usascaled)
model.biplot(label=True, legend=False)

model.biplot3d(label=True, legend=False)
