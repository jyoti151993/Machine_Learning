# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 17:18:43 2023

@author: dbda-lab
"""

import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import recall_score
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import seaborn as sns


comp_prob=pd.read_csv("comp_prob.csv")
print(comp_prob,"\n")

## roc_auc_score--> calculate the values required for drawing ROC curve

### calculate the area under the ROC Curve

###With plot model2 is better 

fpr, tpr, thres=roc_curve(comp_prob['y_test'],comp_prob['yprob_1'])

plt.plot(fpr,tpr)
plt.xlabel("1-Specificity")
plt.ylabel("sensitivity")
plt.show()

print(roc_auc_score(comp_prob["y_test"],["yprob_1"]))


fpr, tpr, thres=roc_curve(comp_prob['y_test'],comp_prob['yprob_2'])


print(roc_auc_score(comp_prob["y_test"],comp_prob["yprob_2"]))



## logloss

print(log_loss(comp_prob["y_test"],comp_prob["yprob_1"]))


print(log_loss(comp_prob["y_test"],comp_prob["yprob_2"]))


import numpy as np

y_test=np.array([14,18,92,43])

y_pred=np.array([13.4,20.9,100.1,40.3])

pred_error=y_test-y_pred
print(pred_error)
mae=np.mean(np.absolute(pred_error))
print("MAE",mae)
mse=np.mean(pred_error**2)
print(mse)

rmse=np.sqrt(mse)
print(rmse)

y_mean=np.mean(y_test)
print(y_mean)

r2 = 1-((np.sum(pred_error**2))/(np.sum((y_test-y_mean)**2)))
print(r2)
 

##### SKLEARN ###########################
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import  r2_score


print(mean_absolute_error(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))
print(r2_score(y_test,y_pred))


