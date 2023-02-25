# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 11:53:14 2023

@author: dbda-lab
"""

import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV,cross_val_score,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")

import warnings
warnings.filterwarnings("ignore")
import os
from sklearn.metrics import r2_score
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")


############ concrete

concrete=pd.read_csv("Concrete_Data.csv")
x=concrete.drop('Strength',axis=1)
y=concrete['Strength']

kfold = KFold(n_splits=5, shuffle=True, random_state=2023)
rf=RandomForestRegressor(random_state=2023)
params={'max_features':[3,4,5,6,7],'n_estimators':[25,50,75]}
gcv=GridSearchCV(rf, param_grid=params, cv=kfold, scoring='r2')
gcv.fit(x,y)
print(gcv.best_params_) # max_feat=4, n_est=75
print(gcv.best_score_) #.911

best_model = gcv.best_estimator_
plt.figure(figsize=(20, 5))

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



##################### Medical  insurance
ins=pd.read_csv("insurance.csv")
dum_ins=pd.get_dummies(ins,drop_first=True)
x=dum_ins.drop("charges",axis=1)
y=dum_ins["charges"]

kfold = KFold(n_splits=5, shuffle=True, random_state=2023)
rf=RandomForestRegressor(random_state=2023)
params={'max_features':[3,4,5,6,7],'n_estimators':[25,50,75]}
gcv=GridSearchCV(rf, param_grid=params, cv=kfold, scoring='r2')
gcv.fit(x,y)
print(gcv.best_params_) # max_feat=3, n_est=75
print(gcv.best_score_) #.843

best_model = gcv.best_estimator_
plt.figure(figsize=(20, 5))

#####feature Importances graph 
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

