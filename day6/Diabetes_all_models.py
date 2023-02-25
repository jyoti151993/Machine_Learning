# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 17:23:49 2023

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
from sklearn.metrics import roc_auc_score, roc_curve, log_loss
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline


# KNN classifier, Linear_disr, Quadratic_disc
# Gaussian Naive Based, Logistic Regression
# scoring =Neg_Log_loss
diab=pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
x=diab.drop("Diabetes_binary", axis=1)
y=diab['Diabetes_binary']

x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,random_state=2023,test_size=0.15)


kfold=StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
lr=LogisticRegression()
params={'penalty':['l1','l2','elastic',None]}
gcv=GridSearchCV(lr, param_grid=params, cv=kfold, scoring='roc_auc')
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)

############ Generate predictions
## KNN 
ks=np.arange(1,11,2)
scaler=StandardScaler()
knn=KNeighborsClassifier()
pipe=Pipeline([("STD",scaler),('KNN',knn)]) 
params={'KNN__n_neighbors':ks} #for accessing n_neigbors attr which is a part of knn and knn is a part of pipe(pipe->knn->n_neigbors)
#knn=KNeighborsClassifier()
gcv=GridSearchCV(pipe, param_grid=params, cv=kfold,verbose=2, scoring='roc_auc')
gcv.fit(x,y)
pd_cv=pd.DataFrame(gcv.cv_results_)
print(gcv.best_params_)
print(gcv.best_score_)


#############
ks=[1,3,5,7,9,11,13,15,17,19,21,23]
scores=[]
for i in ks:
# Using KNN

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


## Gaussian NB 
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2023)
nb=GaussianNB()
results=cross_val_score(nb,x,y,scoring='roc_auc', cv=kfold)
print(results.mean())

########## LDA 

lda=LinearDiscriminantAnalysis()
lda.fit(x_train,y_train)

y_pred=lda.predict(x_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob=lda.predict_proba(x_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))
print(log_loss(y_test,y_pred_prob))


################## K fold Cv #################

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2023)
results=cross_val_score(lda,x,y,scoring='roc_auc', cv=kfold)
print(results.mean())


############ QDA#############
qda=QuadraticDiscriminantAnalysis()
qda.fit(x_train,y_train)

y_pred=qda.predict(x_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob=qda.predict_proba(x_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))
print(log_loss(y_test,y_pred_prob))

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2023)
results=cross_val_score(qda,x,y,scoring='roc_auc', cv=kfold)
print(results.mean()) 