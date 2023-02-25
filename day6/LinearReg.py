# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 10:11:06 2023

@author: dbda-lab
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")

pizza=pd.read_csv('pizza.csv')
lr=LinearRegression()
X=pizza[['Promote']]
y=pizza['Sales']
lr.fit(X,y)


# find coefficients for this fit
print(lr.coef_) 
print(lr.intercept_)

y_pred=lr.predict(X)



########### Insure
insure=pd.read_csv("Insure_auto.csv", index_col=0)
y=insure['Operating_Cost']
X=insure.drop('Operating_Cost', axis=1)

lr.fit(X,y)
print(lr.coef_)
print(lr.intercept_)


# let us assume ---
hm=100
aut=200
pred1=-10084.213130948774+167.32668857*hm+54.10529229*aut

## if we increase the home by 1
hm=101
aut=200
pred2=-10084.213130948774+167.32668857*hm+54.10529229*aut
`
tst=np.array([[100,200],[300,400]])
lr.predict(tst)


################# boston data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score

boston=pd.read_csv("Boston.csv")
y=boston["medv"]
x=boston.drop('medv', axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=2023,test_size=0.3)

lr=LinearRegression()
lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)
print(r2_score(y_test, y_pred))

################ KFold Cv ############

kfold=KFold(n_splits=5, shuffle=True, random_state=2023)
lr=LinearRegression()
results=cross_val_score(lr,x,y, cv=kfold)
print(results.mean())

########## KNN Regression
# Grid Search CV n_neighbor 
## Best_score, Best_params

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


kfold = KFold(n_splits=5, shuffle=True, random_state=2023)
scaler = StandardScaler()
knn = KNeighborsRegressor()
pipe = Pipeline([('STD',scaler),('KNN', knn)])
Ks = np.arange(1,11,1)
params = {'KNN__n_neighbors':Ks}
gcv = GridSearchCV(pipe, param_grid=params, cv=kfold, scoring='r2')
gcv.fit(x, y)
print(gcv.best_params_)
print(gcv.best_score_)

######################### Concrete Dataset  Grid search
import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


concrete=pd.read_csv("Concrete_Data.csv")
x=concrete.drop('Strength',axis=1)
y=concrete['Strength']


kfold=KFold(n_splits=5, shuffle=True, random_state=2023)
scaler=StandardScaler()
knn=KNeighborsRegressor()
pipe=Pipeline([("STD",scaler),('KNN',knn)])
ks=np.arange(1,23,2)
params={'KNN__n_neighbors':ks}
gcv = GridSearchCV(pipe, param_grid=params, cv=kfold, scoring='r2')

gcv.fit(x, y)
pd_cv = pd.DataFrame(gcv.cv_results_)
print(gcv.best_params_)
print(gcv.best_score_)

########## lR Regression Kfold CV
kfold=KFold(n_splits=5, shuffle=True, random_state=2023)
lr=LinearRegression()
results=cross_val_score(lr,x,y, cv=kfold)
print(results.mean())