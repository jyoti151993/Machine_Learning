# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 14:22:51 2023

@author: dbda-lab
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")

pizza=pd.read_csv('pizza.csv')
lr=LinearRegression()
X=pizza[['Promote']]
y=pizza['Sales']

poly=PolynomialFeatures(degree=2)
X_poly=poly.fit_transform(X)
print(poly.get_feature_names_out())

lr.fit(X_poly,y)
print(lr.coef_)


########## Boston
###### KFOLD with different degrees for the polynomial expression 
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

boston=pd.read_csv("Boston.csv")
y=boston["medv"]
X=boston.drop('medv', axis=1)


######## degree =1
poly=PolynomialFeatures(degree=1)
X_poly=poly.fit_transform(X)
print(poly.get_feature_names_out())

lr.fit(X_poly,y)
print(lr.coef_)

kfold=KFold(n_splits=5, shuffle=True, random_state=2023)
lr=LinearRegression()
results=cross_val_score(lr,X_poly,y, cv=kfold)
print(results.mean()) 


######## degree =2
poly=PolynomialFeatures(degree=2)
X_poly=poly.fit_transform(X)
print(poly.get_feature_names_out())

lr.fit(X_poly,y)
print(lr.coef_)

kfold=KFold(n_splits=5, shuffle=True, random_state=2023)
lr=LinearRegression()
results=cross_val_score(lr,X_poly,y, cv=kfold)
print(results.mean())

######## degree =3
poly=PolynomialFeatures(degree=3)
X_poly=poly.fit_transform(X)
print(poly.get_feature_names_out())

lr.fit(X_poly,y)
print(lr.coef_)
kfold=KFold(n_splits=5, shuffle=True, random_state=2023)
lr=LinearRegression()
results=cross_val_score(lr,X_poly,y, cv=kfold)
print(results.mean())


######## degree =4
poly=PolynomialFeatures(degree=4)
X_poly=poly.fit_transform(X)
print(poly.get_feature_names_out())
lr.fit(X_poly,y)
print(lr.coef_)
kfold=KFold(n_splits=5, shuffle=True, random_state=2023)
lr=LinearRegression()
results=cross_val_score(lr,X_poly,y, cv=kfold)
print(results.mean())

############ Grid Search
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

poly=PolynomialFeatures()
lr=LinearRegression()
pipe=Pipeline([("POLY",poly),('LIN',lr)])
params={'POLY__degree':[1,2,3,4]}
gcv=GridSearchCV(pipe, param_grid=params, cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)


############ Concrete

concrete=pd.read_csv("Concrete_Data.csv")
x=concrete.drop('Strength',axis=1)
y=concrete['Strength']

poly=PolynomialFeatures()
lr=LinearRegression()
pipe=Pipeline([("POLY",poly),('LIN',lr)])
params={'POLY__degree':[1,2,3,4]}
gcv=GridSearchCV(pipe, param_grid=params, cv=kfold)
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)
