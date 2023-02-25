# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 18:00:43 2023

@author: dbda-lab
"""

from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")

milk= pd.read_csv("milk.csv", index_col=0)
scaler= StandardScaler()
milkscaled=scaler.fit_transform(milk)

#calculate the linkage: merging
mergings = linkage(milkscaled, method='average')
plt.figure(figsize=(12,8))


dendrogram(mergings, labels=np.array(milk.index),leaf_rotation=45, leaf_font_size=10)
plt.show()


#calculate the linkage: merging
mergings = linkage(milkscaled, method='single')
plt.figure(figsize=(12,8))


dendrogram(mergings, labels=np.array(milk.index),leaf_rotation=45, leaf_font_size=10)
plt.show()
