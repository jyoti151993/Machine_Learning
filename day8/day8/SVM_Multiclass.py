# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 17:29:40 2023

@author: dbda-lab
"""

import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")
import pandas as pd
from sklearn.model_selection import GridSearchCV
from  sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np
from  sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")

image_seg=pd.read_csv("Image_Segmention.csv")

x=image_seg.drop('Class',axis=1)
y=image_seg['Class']

le=LabelEncoder()
le_y=le.fit_transform(y)

## with pipeleine linear
svm=SVC(kernel='linear',probability=True,random_state=2023)
scaler=MinMaxScaler()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
pipe_svm=Pipeline([('SCL',scaler),('SVM',svm)])
params={'SVM__C':np.linspace(0.1,10,20),'SVM__decision_function_shape':["ovo","ovr"]}
gcv=GridSearchCV(pipe_svm, param_grid=params, cv=kfold, scoring='neg_log_loss')
gcv.fit(x,le_y)
print(gcv.best_params_)
print(gcv.best_score_)



########## Polynomial Kernel
from sklearn.preprocessing import MinMaxScaler
svm=SVC(kernel='poly',probability=True,random_state=2023)
scaler=MinMaxScaler()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
pipe_svm=Pipeline([('STD',scaler),('SVM',svm)])
params={'SVM__C':np.linspace(0.1,10,20),'SVM__degree':[2,3,4],'SVM__coef0':np.linspace(0,10,20),'SVM__decision_function_shape':["ovo","ovr"]}
gcv=GridSearchCV(pipe_svm, param_grid=params, cv=kfold,verbose=3, scoring='neg_log_loss')
gcv.fit(x,le_y)
print(gcv.best_params_)
print(gcv.best_score_)


######### Radial Kernel
# verbose is for seeing the execution time
svm=SVC(kernel='rbf',probability=True,random_state=2023)
scaler=MinMaxScaler()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
pipe_svm=Pipeline([('STD',scaler),('SVM',svm)])
params={'SVM__C':np.linspace(0.1,10,20),'SVM__gamma':np.linspace(0,10,20),'SVM__decision_function_shape':["ovo","ovr"]}
gcv=GridSearchCV(pipe_svm, param_grid=params, cv=kfold,verbose=3, scoring='neg_log_loss')
gcv.fit(x,le_y)
print(gcv.best_params_)
print(gcv.best_score_)

################## vehicle Silh #######################

vehicle = pd.read_csv("vehicle.csv")

x = vehicle.drop('Class', axis=1)
y = vehicle['Class']

# Data Partitioning

le = LabelEncoder()
le_y = le.fit_transform(y)
x = vehicle.drop('Class', axis=1)
y = vehicle['Class']

### liner
svm=SVC(kernel='linear',probability=True,random_state=2023)
scaler=MinMaxScaler()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
pipe_svm=Pipeline([('SCL',scaler),('SVM',svm)])
params={'SVM__C':np.linspace(0.1,10,20),'SVM__decision_function_shape':["ovo","ovr"]}
gcv=GridSearchCV(pipe_svm, param_grid=params, cv=kfold, scoring='neg_log_loss')
gcv.fit(x,le_y)
print(gcv.best_params_)
print(gcv.best_score_)


########## Polynomial Kernel
from sklearn.preprocessing import MinMaxScaler
svm=SVC(kernel='poly',probability=True,random_state=2023)
scaler=MinMaxScaler()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
pipe_svm=Pipeline([('STD',scaler),('SVM',svm)])
params={'SVM__C':np.linspace(0.1,10,20),'SVM__degree':[2,3,4],'SVM__coef0':np.linspace(0,10,20),'SVM__decision_function_shape':["ovo","ovr"]}
gcv=GridSearchCV(pipe_svm, param_grid=params, cv=kfold,verbose=3, scoring='neg_log_loss')
gcv.fit(x,le_y)
print(gcv.best_params_)
print(gcv.best_score_)

######### Radial Kernel
# verbose is for seeing the execution time
svm=SVC(kernel='rbf',probability=True,random_state=2023)
scaler=MinMaxScaler()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
pipe_svm=Pipeline([('STD',scaler),('SVM',svm)])
params={'SVM__C':np.linspace(0.1,10,20),'SVM__gamma':np.linspace(0,10,20),'SVM__decision_function_shape':["ovo","ovr"]}
gcv=GridSearchCV(pipe_svm, param_grid=params, cv=kfold,verbose=3, scoring='neg_log_loss')
gcv.fit(x,le_y)
print(gcv.best_params_)
print(gcv.best_score_)



##################### Breast Cancer

breastCancer=pd.read_csv("BreastCancer.csv")

dum_can=pd.get_dummies(breastCancer,drop_first=True)
X=dum_can.drop(['Code','Class_Malignant'],axis=1)
y=dum_can['Class_Malignant']

########## with Min Mx scaler
from sklearn.preprocessing import MinMaxScaler
svm=SVC(kernel='linear')
scaler=MinMaxScaler()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
pipe_svm=Pipeline([('STD',scaler),('SVM',svm)])
params={'SVM__C':np.linspace(0.1,10,5)}
gcv=GridSearchCV(pipe_svm, param_grid=params, cv=kfold, scoring='roc_auc')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)


########## Polynomial Kernel
from sklearn.preprocessing import MinMaxScaler
svm=SVC(kernel='poly')
scaler=MinMaxScaler()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
pipe_svm=Pipeline([('STD',scaler),('SVM',svm)])
params={'SVM__C':np.linspace(0.1,10,5),'SVM__degree':[2,3,4],'SVM__coef0':np.linspace(0,10,20)}
gcv=GridSearchCV(pipe_svm, param_grid=params, cv=kfold, scoring='roc_auc')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)


######### Radial Kernel
svm=SVC(kernel='rbf')
scaler=MinMaxScaler()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
pipe_svm=Pipeline([('STD',scaler),('SVM',svm)])
params={'SVM__C':np.linspace(0.1,10,5),'SVM__gamma':np.linspace(0,10,20)}
gcv=GridSearchCV(pipe_svm, param_grid=params, cv=kfold, scoring='roc_auc')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)


############## IBM Attrition

ibm_hr=pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
dum_ibm=pd.get_dummies(ibm_hr, drop_first=True)
x=dum_ibm.drop('Attrition_Yes', axis=1)
y=dum_ibm['Attrition_Yes']


########## with Min Mx scaler
from sklearn.preprocessing import MinMaxScaler
svm=SVC(kernel='linear')
scaler=MinMaxScaler()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
pipe_svm=Pipeline([('STD',scaler),('SVM',svm)])
params={'SVM__C':np.linspace(0.01,10,6)}
gcv=GridSearchCV(pipe_svm, param_grid=params, cv=kfold, scoring='roc_auc')
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)


########## Polynomial Kernel
from sklearn.preprocessing import MinMaxScaler
svm=SVC(kernel='poly')
scaler=MinMaxScaler()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
pipe_svm=Pipeline([('STD',scaler),('SVM',svm)])
params={'SVM__C':np.linspace(0.01,10,6),'SVM__degree':[2,3,4],'SVM__coef0':np.linspace(0,10,20)}
gcv=GridSearchCV(pipe_svm, param_grid=params, cv=kfold,verbose=3, scoring='roc_auc')
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)


######### Radial Kernel
svm=SVC(kernel='rbf')
scaler=MinMaxScaler()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
pipe_svm=Pipeline([('STD',scaler),('SVM',svm)])
params={'SVM__C':np.linspace(0.01,10,6),'SVM__gamma':np.linspace(0.01,10,6)}
gcv=GridSearchCV(pipe_svm, param_grid=params, cv=kfold,verbose =3 ,scoring='roc_auc')
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_) 
