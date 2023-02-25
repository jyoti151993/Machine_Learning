# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:52:47 2023

@author: dbda-lab
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import os
import numpy as np

os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")


brupt=pd.read_csv("Bankruptcy1.csv")
x=brupt.drop(['NO','D',"YR"],axis=1)
y=brupt['D']

#############
kfold=StratifiedKFold(n_splits=5,shuffle=True, random_state=2023)
lr=LogisticRegression()
params={'penalty':['l1','l2','elastic',None]}
gcv=GridSearchCV(lr, param_grid=params, cv=kfold, scoring='roc_auc')
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)


######## Santander_satisfaction
train=pd.read_csv("train.csv", index_col=0)
X_train=train.drop("TARGET",axis=1)
y_train=train['TARGET']

X_test=pd.read_csv("test.csv",index_col=0)

kfold=StratifiedKFold(n_splits=5,shuffle=True, random_state=2023)
lr=LogisticRegression()
params={'penalty':['l1','l2','elastic',None]}
gcv=GridSearchCV(lr, param_grid=params, cv=kfold, verbose=2,scoring='roc_auc')
gcv.fit(X_train,y_train)
print(gcv.best_params_)
print(gcv.best_score_)

########## Image- Seg ########

from sklearn.preprocessing import LabelEncoder
image_seg=pd.read_csv("Image_Segmention.csv")

x=image_seg.drop('Class',axis=1)
y=image_seg['Class']


le=LabelEncoder()
le_y=le.fit_transform(y)

kfold=StratifiedKFold(n_splits=5,shuffle=True, random_state=2023)
lr=LogisticRegression()
params={'penalty':['l1','l2','elastic',None], 'multi_class':['ovr','multinomial']}
gcv=GridSearchCV(lr, param_grid=params, cv=kfold, verbose=2,scoring='neg_log_loss')
gcv.fit(x,le_y)
print(gcv.best_params_)
print(gcv.best_score_)

############## Diabetes######################