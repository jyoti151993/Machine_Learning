# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 18:51:40 2023

@author: dbda-lab
"""

import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


brupt=pd.read_csv("Bankruptcy1.csv")
x=brupt.drop(['NO','D','YR'],axis=1)
y=brupt['D']

x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,random_state=2023,test_size=0.3)


nb=GaussianNB()
nb.fit(x_train,y_train)
y_pred_prob=nb.predict_proba(x_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))
print(log_loss(y_test,y_pred_prob))

y_pred=nb.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))



########################################## KFOLD CV #############################



kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2023)
nb=GaussianNB()
results=cross_val_score(nb,x,y,scoring='roc_auc', cv=kfold)
print(results.mean())


