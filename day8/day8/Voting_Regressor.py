# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 18:10:54 2023

@author: dbda-lab
"""


import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV,cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.ensemble import VotingRegressor

from sklearn.tree import DecisionTreeRegressor
import numpy as np

import warnings
warnings.filterwarnings("ignore")
import os
from sklearn.metrics import r2_score
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")


housing=pd.read_csv("Housing.csv")
dum_hous=pd.get_dummies(housing, drop_first=True)

x=dum_hous.drop('price',axis=1)
y=dum_hous['price']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=2023, test_size=0.3)

lr=LinearRegression()
elastic=ElasticNet()
dtr=DecisionTreeRegressor(random_state=2023)
voting=VotingRegressor([('LR',lr),("ELASTIC",elastic),("DTR",dtr)])
voting.fit(x_train,y_train)
y_pred=voting.predict(x_test)

print(r2_score(y_test, y_pred))

########## Kfolds with CV
voting=VotingRegressor([('LR',lr),("ELASTIC",elastic),("DTR",dtr)])
kfold =KFold(n_splits=5, shuffle=True, random_state=2023)
results=cross_val_score(voting,x,y,cv=kfold)
print(results.mean())

############ Voting Regressor with weights 

voting=VotingRegressor([('LR',lr),("ELASTIC",elastic),("DTR",dtr)], weights=[67,54,23])
kfold =KFold(n_splits=5, shuffle=True, random_state=2023)
results=cross_val_score(voting,x,y,cv=kfold)
print(results.mean())


######  R2 score for decision tree
dtr=DecisionTreeRegressor(random_state=2023)
kfold =KFold(n_splits=5, shuffle=True, random_state=2023)
results=cross_val_score(dtr,x,y,cv=kfold)
r2_dtr=results.mean()
print(r2_dtr)
 #.3169

####
elastic=ElasticNet()
kfold =KFold(n_splits=5, shuffle=True, random_state=2023)
results=cross_val_score(elastic,x,y,cv=kfold)
r2_el=results.mean()
print(r2_el)
# .559

###
lr=LinearRegression()
kfold =KFold(n_splits=5, shuffle=True, random_state=2023)
results=cross_val_score(lr,x,y,cv=kfold)
r2_lr=results.mean()
print(r2_lr)
#.6439


############ Voting Regressor with weights 

voting=VotingRegressor([('LR',lr),("ELASTIC",elastic),("DTR",dtr)], weights=[r2_lr,r2_el,r2_dtr])
kfold =KFold(n_splits=5, shuffle=True, random_state=2023)
results=cross_val_score(voting,x,y,cv=kfold)
print(results.mean())

##### Grid Search cv
print(voting.get_params())
voting=VotingRegressor([('LR',lr),("ELASTIC",elastic),("DTR",dtr)], weights=[r2_lr,r2_el,r2_dtr])
kfold =KFold(n_splits=5, shuffle=True, random_state=2023)
params={'DTR__max_depth':[None,3,4,5],'DTR__min_samples_split':[2,5,10],'DTR__min_samples_leaf':[1,4,10],'ELASTIC__alpha':np.linspace(0,10,5),'ELASTIC__l1_ratio':np.linspace(0,1,5)}
gcv=GridSearchCV(voting, param_grid=params, cv=kfold)
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)



############## Chemical Process

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Chemical Process Data")

chem=pd.read_csv("ChemicalProcess.csv")
x=chem.drop('Yield',axis=1)
y=chem['Yield']

########## with mean imputation ## 0.335

imputer=SimpleImputer(strategy='mean')
dtr=DecisionTreeRegressor(random_state=2023)
pipe=Pipeline([('IMPUTER',imputer),('TREE',dtr)])
print(pipe.get_params())
params={'TREE__max_depth':[None,4,6],'TREE__min_samples_split':np.arange(2,16,2),'TREE__min_samples_leaf':np.arange(1,16,3)}
kfold =KFold(n_splits=5, shuffle=True, random_state=2023)
gcv=GridSearchCV(pipe, param_grid=params, cv=kfold, verbose=3,n_jobs=-1)
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)

############## with median imputaion ## 0.39
imputer=SimpleImputer(strategy='median')
dtr=DecisionTreeRegressor(random_state=2023)
pipe=Pipeline([('IMPUTER',imputer),('TREE',dtr)])
print(pipe.get_params())
params={'TREE__max_depth':[None,4,6],'TREE__min_samples_split':np.arange(2,16,2),'TREE__min_samples_leaf':np.arange(1,16,3)}
kfold =KFold(n_splits=5, shuffle=True, random_state=2023)
gcv=GridSearchCV(pipe, param_grid=params, cv=kfold, verbose=3,n_jobs=-1)
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)

################### with mean & median together as a params in GCV
imputer=SimpleImputer()
dtr=DecisionTreeRegressor(random_state=2023)
pipe=Pipeline([('IMPUTER',imputer),('TREE',dtr)])
print(pipe.get_params())
params={'TREE__max_depth':[None,4,6],'TREE__min_samples_split':np.arange(2,16,2),'TREE__min_samples_leaf':np.arange(1,16,3),'IMPUTER__strategy':['mean','median']}
kfold =KFold(n_splits=5, shuffle=True, random_state=2023)
gcv=GridSearchCV(pipe, param_grid=params, cv=kfold, verbose=3,n_jobs=-1)
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)

################ Linear regression with Imputer Computation #########
imputer=SimpleImputer()
lr=LinearRegression()
pipe=Pipeline([('IMPUTER',imputer),('LR',lr)])
print(pipe.get_params())
params={'IMPUTER__strategy':['mean','median']}
kfold =KFold(n_splits=5, shuffle=True, random_state=2023)
gcv=GridSearchCV(pipe, param_grid=params, cv=kfold, verbose=3,n_jobs=-1)
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)

################ KNN Imputer
from sklearn.impute import KNNImputer
imputer=KNNImputer()
dtr=DecisionTreeRegressor(random_state=2023)
pipe=Pipeline([('IMPUTER',imputer),('TREE',dtr)])
params={'TREE__max_depth':[None,4,6],'TREE__min_samples_split':np.arange(2,16,2),'TREE__min_samples_leaf':np.arange(1,16,3),'IMPUTER__n_neighbors':[1,2,3,4]}
kfold =KFold(n_splits=5, shuffle=True, random_state=2023)
gcv=GridSearchCV(pipe, param_grid=params, cv=kfold, verbose=3,n_jobs=-1)
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)
