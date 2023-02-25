# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 08:50:31 2023

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

import numpy as np
from sklearn.metrics import log_loss
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

brupt=pd.read_csv("Bankruptcy1.csv")
x=brupt.drop(['NO','D',"YR"],axis=1)
y=brupt['D']

x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,random_state=2023,test_size=0.3)


############ Linear Discriminant analysis ##########
lda=LinearDiscriminantAnalysis()
lda.fit(x_train,y_train)

y_pred=lda.predict(x_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob=lda.predict_proba(x_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))
print(log_loss(y_test,y_pred_prob))


################## K fold Cv #################

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2023)
results=cross_val_score(lda,x,y,scoring='neg_log_loss', cv=kfold)
print(results.mean())


#######################  Quadratic ################## we cal the sigma for every classes
qda=QuadraticDiscriminantAnalysis()
qda.fit(x_train,y_train)

y_pred=qda.predict(x_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob=qda.predict_proba(x_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))
print(log_loss(y_test,y_pred_prob))

############## Kfold CV ###################
 #Scaling is not required we assume its a normal distribution
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2023)
results=cross_val_score(qda,x,y,scoring='neg_log_loss', cv=kfold)
print(results.mean())



##################### Img Segmentation  #######################################

image_seg=pd.read_csv("Image_Segmention.csv")

x=image_seg.drop('Class',axis=1)
y=image_seg['Class']

x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,random_state=2023,test_size=0.3)

le=LabelEncoder()
le_y=le.fit_transform(y)


### linear discriminant analysis
lda=LinearDiscriminantAnalysis()
lda.fit(x_train,y_train)
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2023)
results=cross_val_score(lda,x,le_y,scoring='neg_log_loss', cv=kfold)
print(results.mean())

############  Quadratic ################## we cal the sigma for every classes
qda=QuadraticDiscriminantAnalysis()
qda.fit(x_train,y_train)
######## Kfold CV ###
# Scaling is not required we assume its a normal distribution
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2023)
results=cross_val_score(qda,x,y,scoring='neg_log_loss', cv=kfold)
print(results.mean())


############### Vehicle  identification ###############################

vehicle = pd.read_csv("vehicle.csv")

x = vehicle.drop('Class', axis=1)
y = vehicle['Class']
le = LabelEncoder()
le_y = le.fit_transform(y)

## linear
lda=LinearDiscriminantAnalysis()
lda.fit(x_train,y_train)

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2023)
results=cross_val_score(lda,x,le_y,scoring='neg_log_loss', cv=kfold)
print(results.mean())

############  Quadratic 
qda=QuadraticDiscriminantAnalysis()
qda.fit(x_train,y_train)

###### Kfold CV ################### Scaling is not required we assume its a normal distribution
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2023)
results=cross_val_score(qda,x,y,scoring='neg_log_loss', cv=kfold)
print(results.mean())
