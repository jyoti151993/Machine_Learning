# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 12:21:59 2023

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
import numpy as np

Emp_Sal=pd.read_csv("Exp_Salaries.csv")
dum_Emp=pd.get_dummies(Emp_Sal, drop_first=True)
x=dum_Emp.drop('Salary',axis=1)
y=dum_Emp['Salary']



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