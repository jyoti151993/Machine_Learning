# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 15:34:07 2023

@author: dbda-lab
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from  sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")


############ BankRuptcy with SVM

brupt=pd.read_csv("Bankruptcy1.csv")
x=brupt.drop(['NO','D',"YR"],axis=1)
y=brupt['D']

svm = SVC(kernel= 'linear' ,C=2)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
results= cross_val_score(svm, x, y, cv=kfold, scoring='roc_auc')
print(results.mean())

params={'C':np.linspace(0.1,10,20)}
svm=SVC(kernel='linear')
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
gcv=GridSearchCV(svm, param_grid=params,cv=kfold, scoring='roc_auc' )
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)


########## With Scaling Standard scaler

svm=SVC(kernel='linear')
scaler=StandardScaler()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
pipe_svm=Pipeline([('STD',scaler),('SVM',svm)])
params={'SVM__C':np.linspace(0.1,10,20)}
gcv=GridSearchCV(pipe_svm, param_grid=params, cv=kfold, scoring='roc_auc')
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)


########## with Min Mx scaler
from sklearn.preprocessing import MinMaxScaler
svm=SVC(kernel='linear')
scaler=MinMaxScaler()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
pipe_svm=Pipeline([('STD',scaler),('SVM',svm)])
params={'SVM__C':np.linspace(0.1,10,20)}
gcv=GridSearchCV(pipe_svm, param_grid=params, cv=kfold, scoring='roc_auc')
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)


########## Polynomial Kernel
from sklearn.preprocessing import MinMaxScaler
svm=SVC(kernel='poly')
scaler=MinMaxScaler()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
pipe_svm=Pipeline([('STD',scaler),('SVM',svm)])
params={'SVM__C':np.linspace(0.1,10,20),'SVM__degree':[2,3,4],'SVM__coef0':np.linspace(0,10,20)}
gcv=GridSearchCV(pipe_svm, param_grid=params, cv=kfold, scoring='roc_auc')
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)


######### Radial Kernel
svm=SVC(kernel='rbf')
scaler=MinMaxScaler()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
pipe_svm=Pipeline([('STD',scaler),('SVM',svm)])
params={'SVM__C':np.linspace(0.1,10,20),'SVM__gamma':np.linspace(0,10,20)}
gcv=GridSearchCV(pipe_svm, param_grid=params, cv=kfold, scoring='roc_auc')
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)


############# kyphosis with SVM ############

kyphosis=pd.read_csv("Kyphosis.csv")
dum_kyp =pd.get_dummies(kyphosis,drop_first=True)
X=dum_kyp.drop("Kyphosis_present",axis=1)
y=dum_kyp["Kyphosis_present"]

####### linera min max scaler
svm=SVC(kernel='linear')
scaler=MinMaxScaler()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
pipe_svm=Pipeline([('STD',scaler),('SVM',svm)])
params={'SVM__C':np.linspace(0.1,10,20)}
gcv=GridSearchCV(pipe_svm, param_grid=params, cv=kfold, scoring='roc_auc')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

########## Polynomial Kernel

svm=SVC(kernel='poly')
scaler=MinMaxScaler()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
pipe_svm=Pipeline([('STD',scaler),('SVM',svm)])
params={'SVM__C':np.linspace(0.1,10,20),'SVM__degree':[2,3,4],'SVM__coef0':np.linspace(0,10,20)}
gcv=GridSearchCV(pipe_svm, param_grid=params, cv=kfold, scoring='roc_auc')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)


######### Radial Kernel
svm=SVC(kernel='rbf')
scaler=MinMaxScaler()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
pipe_svm=Pipeline([('STD',scaler),('SVM',svm)])
params={'SVM__C':np.linspace(0.1,10,20),'SVM__gamma':np.linspace(0,10,20)}
gcv=GridSearchCV(pipe_svm, param_grid=params, cv=kfold, scoring='roc_auc')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)