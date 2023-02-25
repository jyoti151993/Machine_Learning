# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 09:21:20 2023

@author: dbda-lab
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold,cross_val_score, GridSearchCV , cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier,StackingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier 


import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")


############################# Bank ruptcy  ####################################
brupt=pd.read_csv("Bankruptcy1.csv")
x=brupt.drop(['NO','D'],axis=1)
y=brupt['D']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, random_state=2023, test_size=0.3)

nb=GaussianNB()
svm_l=SVC(probability=True, random_state=2023,kernel='linear' )
lr=LogisticRegression(random_state=2023,solver='saga')

gbm=GradientBoostingClassifier(random_state=2023)



stack=StackingClassifier(estimators=[('NB',nb),('SVM',svm_l),('LR',lr)],final_estimator=gbm, stack_method='predict_proba')
stack.fit(x_train,y_train)
y_pred=stack.predict(x_test) 
print(accuracy_score(y_test,y_pred))


y_pred_prob=stack.predict_proba(x_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))


################# with passthrough =True, for improving the acuracy and roc_auc_score 
###passthrough = True implies including original columns apart from the predictions in fit and predictions both in final estimator 

nb=GaussianNB()
scaler=MinMaxScaler()
svm_l=SVC(probability=True, random_state=2023,kernel='linear' )
pipe_svm=Pipeline([("MM",scaler),('SVM',svm_l)])
                   
lr=LogisticRegression(random_state=2023, solver='saga')
pipe_l=Pipeline([("MM",scaler),('LR',lr)])
gbm=GradientBoostingClassifier(random_state=2023)

kfold=StratifiedKFold(random_state=2023,shuffle=True)

stack=StackingClassifier(estimators=[('NB',nb),('SVM',pipe_svm),('LR',pipe_l)],final_estimator=gbm,passthrough=True, stack_method='predict_proba')
stack.fit(x_train,y_train)
y_pred=stack.predict(x_test) 
print(accuracy_score(y_test,y_pred))


y_pred_prob=stack.predict_proba(x_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))

####################### With Tunning parameters GridSearch CV
import numpy as np

kfold=StratifiedKFold(random_state=2023,shuffle=True)
stack=StackingClassifier(estimators=[('NB',nb),('SVM',pipe_svm),('LR',pipe_l)],final_estimator=gbm,passthrough=True,cv=kfold, stack_method='predict_proba')
print(stack.get_params())
params={'SVM__SVM__C':np.linspace(.01,10,5),'LR__LR__penalty':['l1','l2','elasticnet',None]}
gcv=GridSearchCV(stack, param_grid=params, cv=kfold, scoring='roc_auc')
gcv.fit(x,y)
print(gcv.best_params_) # {'LR__LR__penalty': 'l2', 'SVM__SVM__C': 7.5024999999999995}
print(gcv.best_score_) #0.8884192730346576


############ Image Seg
from sklearn.ensemble import RandomForestClassifier

image_seg=pd.read_csv("Image_Segmention.csv")

x=image_seg.drop('Class',axis=1)
y=image_seg['Class']

## Data Partitioning  

le=LabelEncoder()
le_y=le.fit_transform(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,random_state=2023,test_size=0.3)

scaler=MinMaxScaler()
svm_l=SVC(probability=True, random_state=2023,kernel='linear' )
pipe_svm=Pipeline([("MM",scaler),('SVM',svm_l)])
dtc=DecisionTreeClassifier(random_state=2023)
nb=GaussianNB()
rf=RandomForestClassifier(random_state=2023)

#stack.fit(x_train,y_train)
#y_pred=stack.predict(x_test) 
#print(accuracy_score(y_test,y_pred))
#y_pred_prob=stack.predict_proba(x_test)[:,1]
#print(roc_auc_scor(y_test,y_pred_prob))
kfold=StratifiedKFold(random_state=2023,shuffle=True)
stack=StackingClassifier(estimators=[('SVM',pipe_svm),('DTC',dtc),('NB',nb)],final_estimator=rf,passthrough=True,cv=kfold, stack_method='predict_proba')
print(stack.get_params())
params={'SVM__SVM__C':np.linspace(0.001,5,10),'DTC__max_depth':[2,4,None],'DTC__min_samples_split':[2,5,10],'DTC__min_samples_leaf':[1,4], 'final_estimator__max_features':[3,4,5,6]}
gcv=GridSearchCV(stack, param_grid=params, cv=kfold, scoring='neg_log_loss',verbose=3, n_jobs=-1)
gcv.fit(x,le_y)
print(gcv.best_params_) #{'DTC__max_depth': None, 'DTC__min_samples_leaf': 1, 'DTC__min_samples_split': 2, 'SVM__SVM__C': 5.0, 'final_estimator__max_features': 5}
print(gcv.best_score_) #-0.28952792377455705
