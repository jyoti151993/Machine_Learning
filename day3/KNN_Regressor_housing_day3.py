# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 11:53:33 2023

@author: dbda-lab
"""


import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import  r2_score
from numpy import np

housing=pd.read_csv("Housing.csv")
dum_hous=pd.get_dummies(housing, drop_first=True)

x=dum_hous.drop('price',axis=1)
y=dum_hous['price']


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=2023,test_size=0.3)



knn=KNeighborsRegressor(n_neighbors=3)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
print(mean_absolute_error(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))
print(r2_score(y_test,y_pred))
 
 
 ################## With Loop #################
    
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=2023,test_size=0.3)


ks=np.arange(1,11)
scores=[]
for i in ks:
 knn=KNeighborsRegressor(n_neighbors=i)
 scaler=StandardScaler()
 pipe=Pipeline([("STD",scaler),('KNN',knn)])
 pipe.fit(x_train,y_train)
 y_pred=pipe.predict(x_test)
 r2=r2_score(y_test,y_pred)
 scores.append(r2)
 print("n_neighbors =",i, "R2= ",r2)
 
 
 
print("Best Score : ",np.max(scores))
i_max=np.argmax(scores)
best_k=ks[i_max]
print("Best parameter=",best_k)