# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 18:00:34 2023

@author: dbda-lab
"""

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")


cancer=pd.read_csv("BreastCancer.csv")
dum_cancer=pd.get_dummies(cancer, drop_first=True)
X=dum_cancer.drop(['Class_Malignant', 'Code'], axis=1)
y=dum_cancer['Class_Malignant']

X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,random_state=2023,test_size=0.3)

scaler=StandardScaler()
prcomp=PCA(n_components=4)
svm=SVC(probability=True, random_state=2023)
pipe=Pipeline([("STD",scaler),('PCA',prcomp),('SVM',svm)])
pipe.fit(X_train,y_train)
y_pred_prob=pipe.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))



############## Grid Search
print(pipe.get_params())
params={'PCA__n_components':[3,4,5,6],'SVM__C':np.linspace(0.01,5,5),'SVM__gamma':np.linspace(0.01,5,5)}

kfold=StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)

gcv=GridSearchCV(pipe, param_grid=params, cv=kfold, verbose=3, n_jobs=-1, scoring='roc_auc')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

########################################### Bank Ruptcy #############

brupt=pd.read_csv("Bankruptcy1.csv")
x=brupt.drop(['NO','D'],axis=1)
y=brupt['D']


x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,random_state=2023,test_size=0.3)

scaler=StandardScaler()
prcomp=PCA(n_components=4)
svm=SVC(probability=True, random_state=2023,kernel='linear')
pipe=Pipeline([("STD",scaler),('PCA',prcomp),('SVM',svm)])
pipe.fit(x_train,y_train)
y_pred_prob=pipe.predict_proba(x_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

###### with Grid Search ###########
print(pipe.get_params())
params={'PCA__n_components':[3,4,5,6,7, 8, 9,10],'SVM__C':np.linspace(0.01,5,5)}

kfold=StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)

gcv=GridSearchCV(pipe, param_grid=params, cv=kfold, verbose=3, n_jobs=-1, scoring='roc_auc')
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)