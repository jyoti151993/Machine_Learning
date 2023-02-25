# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 08:03:10 2023

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

knn=KNeighborsClassifier(n_neighbors=3)
scaler=StandardScaler()
X_scl_trn=scaler.fit_transform(x_train)
knn.fit(X_scl_trn,y_train)


X_scl_tst=scaler.fit_transform(x_test)

y_pred=knn.predict(X_scl_tst)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

#print(y.value_counts(normalize=True)*100)
#print(y_test.value_counts(normalize=True)*100)
#print(y_train.value_counts)




################ Predicted Probabilities
y_pred_prob=knn.predict_proba(X_scl_tst)[:,1]

fpr, tpr, thres=roc_curve(y_test,y_pred_prob)

plt.plot(fpr,tpr)
plt.xlabel("1-Specificity")
plt.ylabel("sensitivity")
plt.show()


print(roc_auc_score(y_test,y_pred_prob))
print(log_loss(y_test,y_pred_prob))

################################# Using Pipeline 
from sklearn.pipeline import Pipeline


knn=KNeighborsClassifier(n_neighbors=3)
scaler=StandardScaler()
pipe=Pipeline([("STD",scaler),('KNN',knn)])

# uneccessary object is not required to created
# to avoid reduncy
# y is already scaled as 0 or 1 so we are just scaling x here 

#X_scl_trn=scaler.fit_transform(x_train)
#knn.fit(X_scl_trn,y_train)
pipe.fit(x_train,y_train) # .fit is equivalent to X_scl_trn=scaler.fit_transform(x_train) &knn.fit(X_scl_trn,y_train)

X_scl_tst=scaler.fit_transform(x_test)

y_pred=pipe.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

#print(y.value_counts(normalize=True)*100)
#print(y_test.value_counts(normalize=True)*100)
#print(y_train.value_counts)


################ Predicted Probabilities

y_pred_prob=pipe.predict_proba(x_test)[:,1]

fpr, tpr, thres=roc_curve(y_test,y_pred_prob)

plt.plot(fpr,tpr)
plt.xlabel("1-Specificity")
plt.ylabel("sensitivity")
plt.show()


print(roc_auc_score(y_test,y_pred_prob))
print(log_loss(y_test,y_pred_prob))