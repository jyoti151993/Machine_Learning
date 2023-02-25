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
from skleran.preprocssing import StandardScaler
from sklearn.pipeline import Pipeline

boston=pd.read_csv("Boston.csv")
y=boston["medv"]
x=boston.drop('medv', axis=1)


####### without scaling 
ridge=Ridge()
kfold = KFold(n_splits=5, shuffle=True, random_state=2023)
params={'alpha':[0.01,0.1,0.5,1,2,3,6,10]}
gcv=GridSearchCV(ridge, param_grid=params, cv=kfold) # by default r2 score in case of rgression
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)
##3 Coefficentt
best_model=gcv.best_estimator_
print(best_model.coef_)

### with scaling
scaler=StandardScaler()
pipe=Pipeline([('STD', scaler),('RIDGE',ridge)])
params={'RIDGE__alpha':[0.01,0.1,0.5,1,2,3,6,10]}
gcv=GridSearchCV(pipe, param_grid=params, cv=kfold)
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)



########## with lasso
lasso=Lasso()
kfold = KFold(n_splits=5, shuffle=True, random_state=2023)
params={'alpha':[0.01,0.1,0.5,1,2,3,6,10]}
gcv=GridSearchCV(lasso, param_grid=params, cv=kfold) # by default r2 score in case of regression
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)
##3 Coefficentt
best_model=gcv.best_estimator_
print(best_model.coef_)


#### with ElasticNet

elastic=ElasticNet()
kfold = KFold(n_splits=5, shuffle=True, random_state=2023)
params={'alpha':[0.01,0.1,0.5,1,2,3,6,10],'l1_ratio':[0,0.25,0.5,0.75,1]}
gcv=GridSearchCV(elastic, param_grid=params, cv=kfold) # by default r2 score in case of regression
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)


############### Concrete

concrete=pd.read_csv("Concrete_Data.csv")
x=concrete.drop('Strength',axis=1)
y=concrete['Strength']

ridge=Ridge()
kfold = KFold(n_splits=5, shuffle=True, random_state=2023)
params={'alpha':[0.01,0.1,0.5,1,2,3,6,10]}
gcv=GridSearchCV(ridge, param_grid=params, cv=kfold) # by default r2 score in case of rgression
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)
##3 Coefficentt
best_model=gcv.best_estimator_
print(best_model.coef_)

########## with lasso
lasso=Lasso()
kfold = KFold(n_splits=5, shuffle=True, random_state=2023)
params={'alpha':[0.01,0.1,0.5,1,2,3,6,10]}
gcv=GridSearchCV(lasso, param_grid=params, cv=kfold) # by default r2 score in case of regression
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)
##3 Coefficentt
best_model=gcv.best_estimator_
print(best_model.coef_)

#### with ElasticNet
elastic=ElasticNet()
kfold = KFold(n_splits=5, shuffle=True, random_state=2023)
params={'alpha':[0.01,0.1,0.5,1,2,3,6,10],'l1_ratio':[0,0.25,0.5,0.75,1]}
gcv=GridSearchCV(elastic, param_grid=params, cv=kfold) # by default r2 score in case of regression
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)