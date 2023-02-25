# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 16:29:40 2023

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


############### Concrete Data Set #############


######################## Unlabelled data  testConcrete######################
tst_conc=pd.read_csv("testConcrete.csv")
scaler=StandardScaler()
knn=KNeighborsRegressor(n_neighbors=3)
pipe=Pipeline([('STD',scaler),('KNN',knn)])

## Build the model on the entire data
#pipe.fit(x,y) # ensures that pipe should look for 19 nearest neighbors in image segmentation dataset
#=pipe.predict(tst_conc)

#predictions=le.inverse_transform(y_pred)
#print(predictions)


############# we dont required the above 4 lines if we are using grid search

### predicting with Grid Search 
y_pred=gcv.predict(tst_conc)
print(y_pred)
