# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 15:55:00 2023

@author: dbda-lab
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold, GridSearchCV,cross_val_score,StratifiedKFold
from sklearn.model_selection import train_test_split

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


############# Concrete  Data set #################

concrete=pd.read_csv("Concrete_Data.csv")
x=concrete.drop('Strength',axis=1)
y=concrete['Strength']

gbr=GradientBoostingRegressor(random_state=2023)
params={'learning_rate':np.linspace(.01,0.6,5),'n_estimators':[15,30,50,75],'max_depth':[2,3,4,5]}
kfold =KFold(n_splits=5, shuffle=True, random_state=2023)
gcv=GridSearchCV(gbr, param_grid=params, cv=kfold)
gcv.fit(x,y)
print(gcv.best_params_) # {'learning_rate': 0.1575, 'max_depth': 5, 'n_estimators': 75}
print(gcv.best_score_) # 0.9276137434938942
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

###########
hgm=HistGradientBoostingRegressor(random_state=2023)
params={'learning_rate':np.linspace(.01,0.6,5),'max_iter':[15,30,50,75],'max_depth':[2,3,4,5]}
print(hgm.get_params())
kfold = KFold(n_splits=5, shuffle=True, random_state=2023)
gcv=GridSearchCV(hgm, param_grid=params, cv=kfold, scoring='r2')
gcv.fit(x,y)
print(gcv.best_params_) #{'learning_rate': 0.305, 'max_depth': 5, 'max_iter': 75}
print(gcv.best_score_) #0.925152028911192
