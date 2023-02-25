# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 07:58:08 2023

@author: dbda-lab
"""
import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score


image_seg=pd.read_csv("Image_Segmention.csv")

x=image_seg.drop('Class',axis=1)
y=image_seg['Class']


le=LabelEncoder()
le_y=le.fit_transform(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,random_state=2023,test_size=0.3)

#We dont need standard scaler in case of gausian naive probab as it is considered that it it follows normal dist bydefault

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2023)
nb=GaussianNB()
results=cross_val_score(nb,x,le_y,scoring='neg_log_loss', cv=kfold)
print(results.mean())





