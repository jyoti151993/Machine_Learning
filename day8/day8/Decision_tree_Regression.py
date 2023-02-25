# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:06:34 2023

@author: dbda-lab
"""

from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import  GridSearchCV,KFold
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")


housing=pd.read_csv("Housing.csv")
dum_hous=pd.get_dummies(housing, drop_first=True)

x=dum_hous.drop('price',axis=1)
y=dum_hous['price']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=2023, test_size=0.3)

dtr = DecisionTreeRegressor(random_state=2023, max_depth=3)
dtr.fit(x_train, y_train)

plt.figure(figsize=(50, 15))
tree.plot_tree(dtr, feature_names=x_train.columns, 
          filled=True, fontsize=22)

y_pred=dtr.predict(x_test)

############# GRID SEARCH #############################

kfold =KFold(n_splits=5, shuffle=True, random_state=2023)
dtr = DecisionTreeRegressor(random_state=2023)
params = {'max_depth': [None, 3, 4, 5, ], 'min_samples_split': [
    2, 5, 10], 'min_samples_leaf': [1, 4, 10]}
gcv = GridSearchCV(dtr, param_grid=params, cv=kfold,
                   verbose=3)
gcv.fit(x, y)
print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_
plt.figure(figsize=(55, 15))
tree.plot_tree(best_model, feature_names=x.columns, 
                filled=True, fontsize=22)

print(best_model.feature_importances_)
print(x.columns)
imps = best_model.feature_importances_
cols = x.columns
plt.barh(cols, imps)
plt.title("feature Importances Plot")
plt.show()

############### Concrete Data Set #############
concrete=pd.read_csv("Concrete_Data.csv")
x=concrete.drop('Strength',axis=1)
y=concrete['Strength']

kfold =KFold(n_splits=5, shuffle=True, random_state=2023)
dtr = DecisionTreeRegressor(random_state=2023)
params = {'max_depth': [None, 2,5 ], 'min_samples_split':np.arange(2,11), 'min_samples_leaf':np.arange(2,11)}
gcv = GridSearchCV(dtr, param_grid=params, cv=kfold,
                   verbose=3)
gcv.fit(x, y)
print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_
plt.figure(figsize=(70, 15))
tree.plot_tree(best_model, feature_names=x.columns, 
                filled=True, fontsize=22)

print(best_model.feature_importances_)
print(x.columns)
imps = best_model.feature_importances_
cols = x.columns
plt.barh(cols, imps)
plt.title("feature Importances Plot")
plt.show()

### fo sorting the bar graph
s_index=np.argsort(imps)
sorted_imps=imps[s_index]
sorted_x=cols[s_index]
plt.barh(sorted_x, sorted_imps)
plt.title("feature Importances Plot")
plt.show()

############ insurance data ###################################################################
ins=pd.read_csv("insurance.csv")
dum_ins=pd.get_dummies(ins,drop_first=True)
x=dum_ins.drop("charges",axis=1)
y=dum_ins["charges"]

kfold =KFold(n_splits=5, shuffle=True, random_state=2023)
dtr = DecisionTreeRegressor(random_state=2023)
params = {'max_depth': [None, 2,5 ], 'min_samples_split':np.arange(2,11), 'min_samples_leaf':np.arange(2,11)}
gcv = GridSearchCV(dtr, param_grid=params, cv=kfold,
                   verbose=3)
gcv.fit(x, y)
print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_
plt.figure(figsize=(70, 15))
tree.plot_tree(best_model, feature_names=x.columns, 
                filled=True, fontsize=22)

print(best_model.feature_importances_)
print(x.columns)
imps = best_model.feature_importances_
cols = x.columns
plt.barh(cols, imps)
plt.title("feature Importances Plot")
plt.show()

### fo sorting the bar graph
s_index=np.argsort(imps)
sorted_imps=imps[s_index]
sorted_x=cols[s_index]
plt.barh(sorted_x, sorted_imps)
plt.title("feature Importances Plot")
plt.show()


################# tabular Aug 2021 dataset #######
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\tabular")

train=pd.read_csv("train.csv",index_col=0)
x=train.drop("loss",axis=1)
y=train["loss"]
test=pd.read_csv("test.csv",index_col=0)
submit=pd.read_csv("sample_submission.csv")


######### Elastic

elastic=ElasticNet()
kfold = KFold(n_splits=5, shuffle=True, random_state=2023)
params={'alpha':np.linspace(0.01,10,5),'l1_ratio':np.linspace(0,1,5)}
gcv=GridSearchCV(elastic, param_grid=params, cv=kfold,verbose=3) # by default r2 score in case of regression
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)


####### without tunning params

dtr = DecisionTreeRegressor(random_state=2023)
dtr.fit(x, y)
y_pred=dtr.predict(test)
submit["loss"]=y_pred
submit.to_csv('sbt_dtr_wo_tunning.csv',index=False)


############

kfold =KFold(n_splits=5, shuffle=True, random_state=2023)
dtr = DecisionTreeRegressor(random_state=2023)
params = {'max_depth': [None, 2,5 ], 'min_samples_split':np.arange(2,11), 'min_samples_leaf':np.arange(2,11)}
gcv = GridSearchCV(dtr, param_grid=params, cv=kfold,
                   verbose=3)
gcv.fit(x, y)
print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_
plt.figure(figsize=(70, 15))
tree.plot_tree(best_model, feature_names=x.columns, 
                filled=True, fontsize=22)

print(best_model.feature_importances_)
print(x.columns)
imps = best_model.feature_importances_
cols = x.columns
plt.barh(cols, imps)
plt.title("feature Importances Plot")
plt.show()

### fo sorting the bar graph
s_index=np.argsort(imps)
sorted_imps=imps[s_index]
sorted_x=cols[s_index]
plt.barh(sorted_x, sorted_imps)
plt.title("feature Importances Plot")
plt.show()
