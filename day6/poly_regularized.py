# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:01:41 2023

@author: dbda-lab
"""
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")

boston=pd.read_csv("Boston.csv")
y=boston["medv"]
X=boston.drop('medv', axis=1)

poly=PolynomialFeatures()
ridge=Ridge()
pipe_ridge=Pipeline([('POLY',poly),('RIDGE',ridge)])
kfold=KFold(n_splits=5, shuffle=True, random_state=True)
params={'RIDGE__alpha':[0.01,0.1,0.5,1,2,3,6,10],'POLY__degree':[1,2,3,4]}
gcv=GridSearchCV(pipe_ridge, param_grid=params, cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)


####### lasso
poly=PolynomialFeatures()
lasso=Lasso()
pipe_lasso=Pipeline([('POLY',poly),('LASSO',lasso)])
kfold=KFold(n_splits=5, shuffle=True, random_state=True)
params={'LASSO__alpha':[0.01,0.1,0.5,1,2,3,6,10],'POLY__degree':[1,2,3,4]}
gcv=GridSearchCV(pipe_lasso, param_grid=params, cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

###### ELASTICNET
poly=PolynomialFeatures()
elas=ElasticNet()
pipe_elast=Pipeline([('POLY',poly),('ELAS',elas)])
kfold=KFold(n_splits=5, shuffle=True, random_state=True)
params={'ELAS__alpha':[0.01,0.1,0.5,1,2,3,6,10],'ELAS__l1_ratio':[0,0.25,0.5,0.75,1],'POLY__degree':[1,2,3,4]}
gcv=GridSearchCV(pipe_elast, param_grid=params, cv=kfold)
gcv.fit(X,y)
pd_elas=pd.DataFrame(gcv.cv_results_)
print(gcv.best_params_)
print(gcv.best_score_)


############## concrete 

concrete=pd.read_csv("Concrete_Data.csv")
x=concrete.drop('Strength',axis=1)
y=concrete['Strength']


########### Ridge
poly=PolynomialFeatures()
ridge=Ridge()
pipe_ridge=Pipeline([('POLY',poly),('RIDGE',ridge)])
kfold=KFold(n_splits=5, shuffle=True, random_state=True)
params={'RIDGE__alpha':[0.01,0.1,0.5,1,2,3,6,10],'POLY__degree':[1,2,3,4]}
gcv=GridSearchCV(pipe_ridge, param_grid=params, cv=kfold)
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)

####### lasso
poly=PolynomialFeatures()
lasso=Lasso()
pipe_lasso=Pipeline([('POLY',poly),('LASSO',lasso)])
kfold=KFold(n_splits=5, shuffle=True, random_state=True)
params={'LASSO__alpha':[0.01,0.1,0.5,1,2,3,6,10],'POLY__degree':[1,2,3,4]}
gcv=GridSearchCV(pipe_lasso, param_grid=params, cv=kfold)
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)

###### ELASTICNET
poly=PolynomialFeatures()
elas=ElasticNet()
pipe_elast=Pipeline([('POLY',poly),('ELAS',elas)])
kfold=KFold(n_splits=5, shuffle=True, random_state=True)
params={'ELAS__alpha':[0.01,0.1,0.5,1,2,3,6,10],'ELAS__l1_ratio':[0,0.25,0.5,0.75,1],'POLY__degree':[1,2,3,4]}
gcv=GridSearchCV(pipe_elast, param_grid=params, cv=kfold)
gcv.fit(x,y)
pd_elas=pd.DataFrame(gcv.cv_results_)
print(gcv.best_params_)
print(gcv.best_score_)
