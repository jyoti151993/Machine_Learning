# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 09:16:09 2023

@author: dbda-lab
"""
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV,cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, classification_report
import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")


# IBM_HR
ibm_hr = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
dum_ibm = pd.get_dummies(ibm_hr, drop_first=True)
x = dum_ibm.drop('Attrition_Yes', axis=1)
y = dum_ibm['Attrition_Yes']


x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, random_state=2023, test_size=0.3)

dtc = DecisionTreeClassifier(random_state=2023, max_depth=3)
dtc.fit(x_train, y_train)

plt.figure(figsize=(35, 10))
tree.plot_tree(dtc, feature_names=x_train.columns, class_names=[
               'No', 'Yes'], filled=True, fontsize=22)


# Test set
y_pred = dtc.predict(x_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

y_pred_prob = dtc.predict_proba(x_test)[:, 1]
print(roc_auc_score(y_test, y_pred_prob))
print(log_loss(y_test, y_pred_prob))

# using Grid Search
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
dtc = DecisionTreeClassifier(random_state=2023)
params = {'max_depth': [None, 3, 4, 5, ], 'min_samples_split': [
    2, 5, 10], 'min_samples_leaf': [1, 4, 10]}
gcv = GridSearchCV(dtc, param_grid=params, cv=kfold,
                   verbose=3, scoring='roc_auc')
gcv.fit(x, y)
print(gcv.best_params_)
print(gcv.best_score_)


# View the best Tree
best_model = gcv.best_estimator_
plt.figure(figsize=(55, 15))
tree.plot_tree(best_model, feature_names=x.columns, class_names=[
               'No', 'Yes'], filled=True, fontsize=22)
print(best_model.feature_importances_)
print(x.columns)
imps = best_model.feature_importances_
cols = x.columns
plt.barh(cols, imps)
plt.tick_params(label_size=100)
plt.title("feature Importances Plot")
plt.show()

########### imag Segmention multi class ##############
image_seg = pd.read_csv("Image_Segmention.csv")

x = image_seg.drop('Class', axis=1)
y = image_seg['Class']

le = LabelEncoder()
le_y = le.fit_transform(y)
print(le.classes_)

# x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,random_state=2023,test_size=0.3)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
dtc = DecisionTreeClassifier(random_state=2023)
params = {'max_depth': [None, 3, 4, 5], 'min_samples_split': [
    2, 5, 10], 'min_samples_leaf': [1, 4, 10]}
gcv = GridSearchCV(dtc, param_grid=params, cv=kfold,
                   verbose=3, scoring='neg_log_loss')
gcv.fit(x, le_y)
print(gcv.best_params_)
print(gcv.best_score_)

# View the best Tree
best_model = gcv.best_estimator_
plt.figure(figsize=(100, 25))
tree.plot_tree(best_model, feature_names=x.columns,
               class_names=le.classes_, filled=True, fontsize=22)

print(best_model.feature_importances_)
print(x.columns)
imps = best_model.feature_importances_
cols = x.columns
plt.barh(cols, imps)
plt.tick_params(label_size=100)
plt.title("feature Importances Plot")
plt.show()

###############  Vehicle Silhouettes #######################

vehicle = pd.read_csv("vehicle.csv")
x = vehicle.drop('Class', axis=1)
y = vehicle['Class']

# Data Partitioning

le = LabelEncoder()
le_y = le.fit_transform(y)
print(le.classes_)


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
dtc = DecisionTreeClassifier(random_state=2023)
params = {'max_depth': [None, 3, 4, 5], 'min_samples_split': [
    2, 4, 6, 8, 10], 'min_samples_leaf': [1, 3, 5, 9, 11]}
gcv = GridSearchCV(dtc, param_grid=params, cv=kfold,
                   verbose=3, scoring='neg_log_loss')
gcv.fit(x, le_y)
print(gcv.best_params_)
print(gcv.best_score_)

# View the best Tree
best_model = gcv.best_estimator_
plt.figure(figsize=(50, 15))
tree.plot_tree(best_model, feature_names=x.columns,
               class_names=le.classes_, filled=True, fontsize=22)
# feature Importances
print(best_model.feature_importances_)
print(x.columns)
imps = best_model.feature_importances_
cols = x.columns
plt.barh(cols, imps)
plt.title("feature Importances Plot")
plt.show()


############################# Bank ruptcy
brupt=pd.read_csv("Bankruptcy1.csv")
x=brupt.drop(['NO','D'],axis=1)
y=brupt['D']

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
dtc = DecisionTreeClassifier(random_state=2023)
params = {'max_depth': [None, 3, 4, 5], 'min_samples_split': [
    2, 4, 6, 8, 10], 'min_samples_leaf': [1, 3, 5, 9, 11]}
gcv = GridSearchCV(dtc, param_grid=params, cv=kfold,
                   verbose=3, scoring='roc_auc')
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)

# View the best Tree
best_model = gcv.best_estimator_
plt.figure(figsize=(60, 20))
tree.plot_tree(best_model, feature_names=x.columns,filled=True, fontsize=22)
# feature Importances'
print(best_model.feature_importances_)
print(x.columns)
imps = best_model.feature_importances_
cols = x.columns
plt.barh(cols, imps)
plt.title("feature Importances Plot")
plt.show()