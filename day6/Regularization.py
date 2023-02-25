# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 12:33:44 2023

@author: dbda-lab
"""
import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

boston=pd.read_csv("Boston.csv")
y=boston["medv"]
x=boston.drop('medv', axis=1)

ridge=Ridge()
kfold = KFold(n_splits=5, shuffle=True, random_state=2023)
params={'alpha':[0.01,0.1,0.5,1,2,3,6,10]}
gcv=GridSearchCV(ridge, param_grid=params, cv=kfold)
gcv.fir(x,y)
print(gcv.best_params_)
print(gcv.best_score_)