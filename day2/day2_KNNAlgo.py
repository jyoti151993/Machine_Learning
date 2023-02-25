# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:56:45 2023

@author: dbda-lab
"""
import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

mowers=pd.read_csv("RidingMowers.csv")
dum_mow=pd.get_dummies(mowers,drop_first=True)
x=dum_mow.drop('Response_Not Bought',axis=1)
y=dum_mow['Response_Not Bought']

x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,random_state=2023,test_size=0.3)

mowers.shape

knn=KNeighborsClassifier(n_neighbors=9)

#X_scl_tst=scaler.fit_transform(x_train)

knn.fit(x_train,y_train)

y_pred=knn.predict(x_train)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

#print(y.value_counts(normalize=True)*100)
#print(y_test.value_counts(normalize=True)*100)

#print(y_train.value_counts)
################ Predicted Probabilities
y_pred_prob=knn.predict_proba(x_test)[:,1]

fpr, tpr, thres=roc_curve(y_test,y_pred_prob)

plt.plot(fpr,tpr)
plt.xlabel("1-Specificity")
plt.ylabel("sensitivity")
plt.show()


print(roc_auc_score(y_test,y_pred_prob))
print(log_loss(y_test,y_pred_prob))
