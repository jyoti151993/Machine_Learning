# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 15:50:17 2023

@author: dbda-lab
"""

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")


vehicle = pd.read_csv("vehicle.csv")

x = vehicle.drop('Class', axis=1)
y = vehicle['Class']

# Data Partitioning

le = LabelEncoder()
le_y = le.fit_transform(y)
x = vehicle.drop('Class', axis=1)
y = vehicle['Class']


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
scaler = StandardScaler()
knn = KNeighborsClassifier()

pipe = Pipeline([("STD", scaler), ('KNN', knn)])
ks = np.arange(1, 23, 2)
# for accessing n_neigbors attr which is a part of knn and knn is a part of pipe(pipe->knn->n_neigbors)
params = {'KNN__n_neighbors': ks}
knn = KNeighborsClassifier()
gcv = GridSearchCV(pipe, param_grid=params, cv=kfold, scoring='neg_log_loss')

gcv.fit(x, y)
pd_cv = pd.DataFrame(gcv.cv_results_)
print(gcv.best_params_)
print(gcv.best_score_)
