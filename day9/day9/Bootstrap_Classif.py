# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 09:10:37 2023

@author: dbda-lab
"""


import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV,cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.ensemble import VotingRegressor , BaggingClassifier
import matplotlib
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import r2_score
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")


################# Bankruptcy ###############################

################ using logistic Regression & bagging ##############

brupt=pd.read_csv("Bankruptcy1.csv")
x=brupt.drop(['NO','D'],axis=1)
y=brupt['D']

x_train, x_test, y_train, y_test = train_test_split (x,y,stratify=y, random_state=2023, test_size=0.3)

lr=LogisticRegression()
bagging=BaggingClassifier(base_estimator=lr, n_estimators=15, random_state=2023)
bagging.fit(x_train,y_train)
y_pred=bagging.predict(x_test)
print(accuracy_score(y_test,y_pred))   ## .825

y_pred_prob=bagging.predict_proba(x_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))    ## 0.875


########### gausian naive basesd algo with bagging ################################################
nb=GaussianNB()
bagging=BaggingClassifier(base_estimator=nb, n_estimators=15, random_state=2023)
bagging.fit(x_train,y_train)
y_pred=bagging.predict(x_test)
print(accuracy_score(y_test,y_pred))  ## .825

y_pred_prob=bagging.predict_proba(x_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))     ## 0.915


################# With Linear Discriminant analysis & Bagging #########

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda=LinearDiscriminantAnalysis()
bagging=BaggingClassifier(base_estimator=lda, n_estimators=15, random_state=2023)
bagging.fit(x_train,y_train)
y_pred=bagging.predict(x_test)
print(accuracy_score(y_test,y_pred))  # .8

y_pred_prob=bagging.predict_proba(x_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))   ##0.9225


########### Grid Search CV #########################
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
bagging=BaggingClassifier(random_state=2023,n_estimators=15, )
print(bagging.get_params())
lr=LogisticRegression()
nb=GaussianNB()
svm_l=SVC(probability=True, random_state=2023, kernel='linear')
svm_r=SVC(probability=True, random_state=2023, kernel='rbf')
dtc=DecisionTreeClassifier(random_state=2023)
lda=LinearDiscriminantAnalysis()
params={'base_estimator':[lr,nb,lda,svm_l,svm_r, dtc]}
gcv=GridSearchCV(bagging, param_grid=params, cv=kfold, scoring='roc_auc')
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)  #0.87844


############## Vehicle Silhouteses ############

########## with GCV cv & bagging 

from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

vehicle = pd.read_csv("vehicle.csv")
x = vehicle.drop('Class', axis=1)
y = vehicle['Class']

# Data Partitioning

le = LabelEncoder()
le_y = le.fit_transform(y)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
bagging=BaggingClassifier(random_state=2023,n_estimators=15, )
print(bagging.get_params())
lr=LogisticRegression()
nb=GaussianNB()
svm_l=SVC(probability=True, random_state=2023, kernel='linear')
svm_r=SVC(probability=True, random_state=2023, kernel='rbf')
dtc=DecisionTreeClassifier(random_state=2023)
lda=LinearDiscriminantAnalysis()
qda=QuadraticDiscriminantAnalysis()
params={'base_estimator':[lr,nb,lda,qda,svm_l,svm_r, dtc]}
gcv=GridSearchCV(bagging, param_grid=params, cv=kfold, scoring='neg_log_loss')
gcv.fit(x,le_y)
print(gcv.best_params_) ## QDA 
print(gcv.best_score_)  #-0.444