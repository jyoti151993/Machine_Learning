# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 16:03:27 2023

@author: dbda-lab
"""

import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


housing=pd.read_csv("Housing.csv")
dum_hous=pd.get_dummies(housing, drop_first=True)

x=dum_hous.drop('price',axis=1)
y=dum_hous['price']


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


############### Concrete Data Set #############