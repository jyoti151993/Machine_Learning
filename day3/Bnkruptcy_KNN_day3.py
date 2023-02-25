# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 08:55:30 2023

@author: dbda-lab
"""

import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

brupt=pd.read_csv("Bankruptcy1.csv")

#dum_bank=pd.get_dummies(bank,drop_first=True)
x=brupt.drop(['NO','D','YR'],axis=1)
y=brupt['D']

x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,random_state=2023,test_size=0.3)

ks=[1,3,5,7,9,11,13]
scores=[]
for i in ks:
# Using Pipeline 

 knn=KNeighborsClassifier(n_neighbors=i)
 scaler=StandardScaler()
 pipe=Pipeline([("STD",scaler),('KNN',knn)])
 pipe.fit(x_train,y_train)
 y_pred_prob=pipe.predict_proba(x_test)[:,1]

 auc=roc_auc_score(y_test,y_pred_prob)
 scores.append(auc)
 
 #print("logloss",log_loss(y_test,y_pred_prob))
 print("n_neighbors =",i,"AUC=",auc)

print("Best Score", np.max(scores))
#best_score=np.max(scores)
#i_max=scores.index(best_score)
i_max=np.argmax(scores)
best_k=ks[i_max]
print("Best parameter =", best_k)
