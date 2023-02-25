# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 10:49:39 2023

@author: dbda-lab
"""


import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV,cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.ensemble import  BaggingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet

from sklearn.tree import DecisionTreeRegressor
import numpy as np

import warnings
warnings.filterwarnings("ignore")
import os
from sklearn.metrics import r2_score
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")

concrete=pd.read_csv("Concrete_Data.csv")
x=concrete.drop('Strength',axis=1)
y=concrete['Strength']


kfold = KFold(n_splits=5, shuffle=True, random_state=2023)
bagging=BaggingRegressor(random_state=2023,n_estimators=15, )
print(bagging.get_params())
lr= LinearRegression()
lasso=Lasso()
ridge=Ridge()
elastic=ElasticNet()
dtr=DecisionTreeRegressor(random_state=2023)
params={'base_estimator':[lr,lasso,ridge,elastic, dtr]}
gcv=GridSearchCV(bagging, param_grid=params, cv=kfold, n_jobs=-1,verbose=3,scoring='r2')
gcv.fit(x,y)
print(gcv.best_params_) ## DecisionTree Regression
print(gcv.best_score_)  #0.9036

############ medical Cost Expenses /insurance.csv ###########

ins=pd.read_csv("insurance.csv")
dum_ins=pd.get_dummies(ins,drop_first=True)
x=dum_ins.drop("charges",axis=1)
y=dum_ins["charges"]

kfold = KFold(n_splits=5, shuffle=True, random_state=2023)
bagging=BaggingRegressor(random_state=2023,n_estimators=15, )
print(bagging.get_params())
lr= LinearRegression()
lasso=Lasso()
ridge=Ridge()
elastic=ElasticNet()
dtr=DecisionTreeRegressor(random_state=2023)
params={'base_estimator':[lr,lasso,ridge,elastic, dtr]}
gcv=GridSearchCV(bagging, param_grid=params, cv=kfold, n_jobs=-1,verbose=3,scoring='r2')
gcv.fit(x,y)
print(gcv.best_params_) ## DecisionTree Regression
print(gcv.best_score_)  #0.826