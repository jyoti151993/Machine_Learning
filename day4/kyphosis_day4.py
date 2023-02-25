# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 14:37:07 2023

@author: dbda-lab
"""

import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd 

kyphosis=pd.read_csv("Kyphosis.csv")
dum_kyp =pd.get_dummies(kyphosis,drop_first=True)
X=dum_kyp.drop("Kyphosis_present",axis=1)
y=dum_kyp["Kyphosis_present"]


kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2023)
scaler=StandardScaler()
knn=KNeighborsClassifier()

pipe=Pipeline([("STD",scaler),('KNN',knn)]) 
ks=np.arange(1,23,2)
params={'KNN__n_neighbors':ks} #for accessing n_neigbors attr which is a part of knn and knn is a part of pipe(pipe->knn->n_neigbors)
knn=KNeighborsClassifier()
gcv=GridSearchCV(pipe, param_grid=params, cv=kfold, scoring='roc_auc')

gcv.fit(X,y)
pd_cv=pd.DataFrame(gcv.cv_results_)
print(gcv.best_params_)
print(gcv.best_score_)


############## Img Segmentations

image_seg=pd.read_csv("Image_Segmention.csv")

x=image_seg.drop('Class',axis=1)
y=image_seg['Class']

## Data Partitioning  

le=LabelEncoder()
le_y=le.fit_transform(y)
x=image_seg.drop('Class',axis=1)
y=image_seg['Class']



kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2023)
scaler=StandardScaler()
knn=KNeighborsClassifier()

pipe=Pipeline([("STD",scaler),('KNN',knn)]) 
ks=np.arange(1,23,2)
params={'KNN__n_neighbors':ks} #for accessing n_neigbors attr which is a part of knn and knn is a part of pipe(pipe->knn->n_neigbors)
knn=KNeighborsClassifier()
gcv=GridSearchCV(pipe, param_grid=params, cv=kfold, scoring='neg_log_loss')

gcv.fit(x,y)
pd_cv=pd.DataFrame(gcv.cv_results_)
print(gcv.best_params_)
print(gcv.best_score_)



################################ Unlabelled data ######################
tst_img=pd.read_csv("tst_img.csv")
scaler=StandardScaler()
knn=KNeighborsClassifier(n_neighbors=19)
pipe=Pipeline([('STD',scaler),('KNN',knn)])

## Build the model on the entire data
pipe.fit(x,le_y) # ensures that pipe should look for 19 nearest neighbors in image segmentation dataset
y_pred=pipe.predict(tst_img)
print(le.classes_)
predictions=le.inverse_transform(y_pred)
print(predictions)


############# we dont required the above 4 lines if we are using grid search

### predicting with Grid Search 
y_pred=gcv.predict(tst_img)
print(y_pred)
