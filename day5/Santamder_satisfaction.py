# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 09:56:38 2023

@author: dbda-lab
"""
import pandas as pd
import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
train=pd.read_csv("train.csv", index_col=0)
X=train.drop("TARGET",axis=1)
y=train['TARGET']




lda=LinearDiscriminantAnalysis()
lda.fit(X,y)

test=pd.read_csv("test.csv")
X_test=test.drop("ID", axis=1)
y_pred_prob=lda.predict_proba(X_test)[:,1]

s_submission=pd.read_csv("Sample_submission.csv")
s_submission['TARGET']=y_pred_prob

s_submission.to_csv("sbt_LDA.csv",index=False)