# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 10:50:47 2023

@author: dbda-lab
"""

import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.pipeline import Pipeline

image_seg=pd.read_csv("Image_Segmention.csv")

x=image_seg.drop('Class',axis=1)
y=image_seg['Class']

## Data Partitioning  

le=LabelEncoder()
le_y=le.fit_transform(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,random_state=2023,test_size=0.3)

ks=[1,3,5,7,9,11,13,15,17,19,21,23]
scores=[]
for i in ks:
 
  # Using Pipeline
 knn=KNeighborsClassifier(n_neighbors=i)
 scaler=StandardScaler()
 pipe=Pipeline([("STD",scaler),('KNN',knn)])
 pipe.fit(x_train,y_train)
 y_pred_prob=pipe.predict_proba(x_test) #including all the columns
 
 ll=log_loss(y_test,y_pred_prob)
 scores.append(ll)
 print("n_neighbors = ",i,"log loss =",ll)
 
print("Min log_loss is" , np.min(scores))


print("Best Score", np.min(ll))
#best_score=np.max(scores)
#i_max=scores.index(best_score)
i_min=np.argmin(scores)
best_k=ks[i_min]
print("Best parameter =", best_k)

