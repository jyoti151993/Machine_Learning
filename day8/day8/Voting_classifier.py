# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 16:53:02 2023

@author: dbda-lab
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV,cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")


##################### Kyphosis Dataset ####################################
kyphosis=pd.read_csv("Kyphosis.csv")
dum_kyp =pd.get_dummies(kyphosis,drop_first=True)
x=dum_kyp.drop("Kyphosis_present",axis=1)
y=dum_kyp["Kyphosis_present"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, random_state=2023, test_size=0.3)

svm=SVC(probability=True, kernel='linear')
nb=GaussianNB()
lda=LinearDiscriminantAnalysis()
voting=VotingClassifier([('SVM',svm),("NB",nb),("LDA",lda)])
voting.fit(x_train,y_train)
y_pred=voting.predict(x_test)
print(accuracy_score(y_test, y_pred))

######### AttributeError: predict_proba is not available when voting='hard' ###############
svm=SVC(probability=True, kernel='rbf')
nb=GaussianNB()
lda=LinearDiscriminantAnalysis()
voting=VotingClassifier([('SVM',svm),("NB",nb),("LDA",lda),],voting='soft')
voting.fit(x_train,y_train)
y_pred=voting.predict(x_test)
print(accuracy_score(y_test, y_pred))
y_pred_prob=voting.predict_proba(x_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))
print(log_loss(y_test, y_pred_prob))

############## rbf
svm=SVC(probability=True, kernel='rbf',random_state=2023)
nb=GaussianNB()
lda=LinearDiscriminantAnalysis()
voting=VotingClassifier([('SVM',svm),("NB",nb),("LDA",lda),],voting='soft')
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
results=cross_val_score(voting,x,y,cv=kfold,scoring='roc_auc')
print(results.mean())

####### linear
svm=SVC(probability=True, kernel='linear',random_state=2023)
nb=GaussianNB()
lda=LinearDiscriminantAnalysis()
voting=VotingClassifier([('SVM',svm),("NB",nb),("LDA",lda),],voting='soft')
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
results=cross_val_score(voting,x,y,cv=kfold,scoring='roc_auc')
print(results.mean())


########## Grid Search with CV
print(voting.get_params())
params={'SVM__C':[0.01,0.1,0.5,1,1.5,2]}
gcv=GridSearchCV(voting, param_grid=params, cv=kfold, scoring='roc_auc')
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)


############## Bankruptcy
from sklearn.linear_model import LogisticRegression
brupt=pd.read_csv("Bankruptcy1.csv")
x=brupt.drop(['NO','D'],axis=1)
y=brupt['D']


#### Models -> GaussianNB, DecisionTree, LogisticRegression
## tune for decision tree 

nb=GaussianNB()
dtc = DecisionTreeClassifier(random_state=2023)
lr=LogisticRegression()
voting=VotingClassifier([('DTC',dtc),("NB",nb),("LR",lr)],voting='soft')
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
print(voting.get_params())
params = {'DTC__max_depth': [None, 3, 4, 5], 'DTC__min_samples_split': [
    2, 4, 6, 8, 10], 'DTC__min_samples_leaf': [1, 3, 5, 9, 11],'LR__penalty':['l1','l2','elasticnet',None]}

gcv=GridSearchCV(voting, param_grid=params, cv=kfold, scoring='roc_auc',n_jobs=-1,verbose=3)
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)