# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 15:16:07 2023

@author: dbda-lab
"""

 from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
 import pandas as pd
 from sklearn.model_selection import StratifiedKFold, GridSearchCV,cross_val_score

 import matplotlib
 from sklearn.tree import DecisionTreeClassifier
 import numpy as np
 from sklearn.metrics import accuracy_score, roc_auc_score
 import warnings
 warnings.filterwarnings("ignore")
 import os
 from sklearn.naive_bayes import GaussianNB
 from sklearn.metrics import r2_score
 
 import os
 os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")
 
 ########### Bank ruptcy #################
 
 brupt=pd.read_csv("Bankruptcy1.csv")
 x=brupt.drop(['NO','D'],axis=1)
 y=brupt['D']


gbm=GradientBoostingClassifier(random_state=2023)
params={'learning_rate':[0.1,0.15,0.3,0.35,0.4,0.5],'n_estimators':[25,50,75],'max_depth':[2,3,4,5]}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
gcv=GridSearchCV(gbm, param_grid=params, cv=kfold, scoring='roc_auc')
gcv.fit(x,y)
print(gcv.best_params_) #{'learning_rate': 0.3, 'max_depth': 2, 'n_estimators': 25}
print(gcv.best_score_)  # 0.8913778529163144
best_model = gcv.best_estimator_
plt.figure(figsize=(30, 15))
print(best_model.feature_importances_)
print(x.columns)
imps = best_model.feature_importances_
cols = x.columns
plt.barh(cols, imps)
plt.title("feature Importances Plot")
plt.show()
plt.figure(figsize=(40, 15))
s_index=np.argsort(imps)
sorted_imps=imps[s_index]
sorted_x=cols[s_index]
plt.barh(sorted_x, sorted_imps)
plt.title("feature Importances Plot")
plt.show()



######### histgrad wth Banking 

hgm=HistGradientBoostingClassifier(random_state=2023)
params={'learning_rate':[0.1,0.15,0.3,0.35,0.4,0.5],'max_iter':[25,50,75],'max_depth':[2,3,4,5]}
print(hgm.get_params())
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
gcv=GridSearchCV(hgm, param_grid=params, cv=kfold, scoring='roc_auc')
gcv.fit(x,y)
print(gcv.best_params_) #{'learning_rate': 0.1, 'max_depth': 2, 'max_iter': 25}
print(gcv.best_score_) # 0.902240067624683 
                         
                                   
                                   
                                   

######################IBM Attrition ##################################
ibm_hr=pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
dum_ibm=pd.get_dummies(ibm_hr, drop_first=True)
x=dum_ibm.drop(['Attrition_Yes','EmployeeNumber'], axis=1)
y=dum_ibm['Attrition_Yes']


gbm=GradientBoostingClassifier(random_state=2023)
params={'learning_rate':[0.1,0.15,0.3,0.35,0.4,0.5],'n_estimators':[25,50,75],'max_depth':[2,3,4,5]}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
gcv=GridSearchCV(gbm, param_grid=params, cv=kfold, scoring='roc_auc')
gcv.fit(x,y)
print(gcv.best_params_) #{'learning_rate': 0.15, 'max_depth': 2, 'n_estimators': 75}
print(gcv.best_score_) # 0.8210384266388966
best_model = gcv.best_estimator_
plt.figure(figsize=(30, 15))
print(best_model.feature_importances_)
print(x.columns)
imps = best_model.feature_importances_
cols = x.columns
plt.barh(cols, imps)
plt.title("feature Importances Plot")
plt.show()
plt.figure(figsize=(40, 15))
s_index=np.argsort(imps)
sorted_imps=imps[s_index]
sorted_x=cols[s_index]
plt.barh(sorted_x, sorted_imps)
plt.title("feature Importances Plot")
plt.show()


########## HistGradientBossting  with IBM Data set
hgm=HistGradientBoostingClassifier(random_state=2023)
params={'learning_rate':[0.1,0.15,0.3,0.35,0.4,0.5],'max_iter':[25,50,75],'max_depth':[2,3,4,5]}
print(hgm.get_params())
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
gcv=GridSearchCV(hgm, param_grid=params, cv=kfold, scoring='roc_auc')
gcv.fit(x,y)
print(gcv.best_params_)  #{'learning_rate': 0.1, 'max_depth': 3, 'max_iter': 50}
print(gcv.best_score_)  #0.8115691110018602
                         