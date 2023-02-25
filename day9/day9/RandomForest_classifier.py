# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 11:32:33 2023

@author: dbda-lab
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV,cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")

############# Bank ruptcy

brupt=pd.read_csv("Bankruptcy1.csv")
x=brupt.drop(['NO','D'],axis=1)
y=brupt['D']


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
rf=RandomForestClassifier(random_state=2023)
params={'max_features':[3,4,5,6,7],'n_estimators':[25,50,75]}
gcv=GridSearchCV(rf, param_grid=params, cv=kfold, scoring='roc_auc')
gcv.fit(x,y)
print(gcv.best_params_) # max_feat=3, n_est=75
print(gcv.best_score_) # 0.9069

best_model = gcv.best_estimator_
plt.figure(figsize=(40, 20))
tree.plot_tree(best_model, feature_names=x.columns,filled=True, fontsize=22)
# feature Importances'
print(best_model.feature_importances_)
print(x.columns)
imps = best_model.feature_importances_
cols = x.columns
plt.barh(cols, imps)
plt.title("feature Importances Plot")
plt.show()
s_index=np.argsort(imps)
sorted_imps=imps[s_index]
sorted_x=cols[s_index]
plt.barh(sorted_x, sorted_imps)
plt.title("feature Importances Plot")
plt.show()

########################## IBM ATTRITION   ###########
ibm_hr=pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
dum_ibm=pd.get_dummies(ibm_hr, drop_first=True)
x=dum_ibm.drop(['Attrition_Yes','EmployeeNumber'], axis=1)
y=dum_ibm['Attrition_Yes']

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
rf=RandomForestClassifier(random_state=2023)
params={'max_features':[3,5,10,15,20],'n_estimators':[25,50,75]}
gcv=GridSearchCV(rf, param_grid=params, cv=kfold,scoring='roc_auc')
gcv.fit(x,y)
print(gcv.best_params_) # max_feat=15, n_est=75
print(gcv.best_score_) #.80446

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
