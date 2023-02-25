# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:25:46 2023

@author: dbda-lab
"""

import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import GridSearchCV

breastCancer=pd.read_csv("BreastCancer.csv")

dum_can=pd.get_dummies(breastCancer,drop_first=True)
X=dum_can.drop(['Code','Class_Malignant'],axis=1)
y=dum_can['Class_Malignant']

#################### Tunning  with Scaling ######################
ks=np.arange(1,21,2)
scores=[]
for i in ks:
   scaler=StandardScaler()
   knn=KNeighborsClassifier(n_neighbors=i)
   pipe=Pipeline([("STD",scaler),('KNN',knn)]) 
   kfold=StratifiedKFold(n_splits=5,shuffle=True, random_state=2023)
   results=cross_val_score(pipe,X,y,cv=kfold,scoring='roc_auc') #by default cross_val_score gives accuracy score  also roc_auc is used for categorical data
   scores.append(results.mean())
   print("n_neighbors =",i, "roc_auc= ",results.mean())
   
    
 
print("Best Score : ",np.max(scores))
i_max=np.argmax(scores)
best_k=ks[i_max]
print("Best parameter=",best_k)


###################### Grid Search CV with Scaling ###################################
scaler=StandardScaler()
knn=KNeighborsClassifier()
pipe=Pipeline([("STD",scaler),('KNN',knn)]) 
params={'KNN__n_neighbors':ks} #for accessing n_neigbors attr which is a part of knn and knn is a part of pipe(pipe->knn->n_neigbors)
knn=KNeighborsClassifier()
gcv=GridSearchCV(pipe, param_grid=params, cv=kfold, scoring='roc_auc')

gcv.fit(X,y)
pd_cv=pd.DataFrame(gcv.cv_results_)
print(gcv.best_params_)
print(gcv.best_score_)