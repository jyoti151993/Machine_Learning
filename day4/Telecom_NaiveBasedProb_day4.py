# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 18:01:01 2023

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
from sklearn.naive_bayes import BernoulliNB
import numpy as np
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline

telecom=pd.read_csv("Telecom.csv")

dum_tel=pd.get_dummies(telecom,drop_first=True)
x=dum_tel.drop("Response_Y",axis=1)
y=dum_tel["Response_Y"]

x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,random_state=2023,test_size=0.3)


nb=BernoulliNB()
nb.fit(x_train,y_train)
y_pred_prob=nb.predict_proba(x_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))
print(log_loss(y_test,y_pred_prob))

y_pred=nb.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))


########################################## KFOLD CV #############################

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2023)
nb=BernoulliNB()
results=cross_val_score(nb,x,y,scoring='roc_auc', cv=kfold)
print(results.mean())