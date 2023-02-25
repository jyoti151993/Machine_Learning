# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 12:28:56 2023

@author: dbda-lab
"""

import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

telecom=pd.read_csv("Telecom.csv")

print(telecom.head())

dum_tel=pd.get_dummies(telecom)#drop_first=True)
print("dum_tel",dum_tel.head())

# with drop
dum_tel1=pd.get_dummies((telecom),drop_first=True)
print("dum_tel1",dum_tel1.head())

#OR
ohe = OneHotEncoder()
hot_encoded=ohe.fit_transform(telecom).toarray()
type(hot_encoded)
print("hot_encoded",hot_encoded)


################## Missing values

job=pd.read_csv("JobSalary2.csv")
print(job)
# Columns having the missing values
job.isnull().sum()>0  

## Removing records with na
job.dropna()


### Constant Imputation
imputer=SimpleImputer(strategy='constant',fill_value=50)
print(imputer.fit_transform(job))


#### Mean Imputation
imputer=SimpleImputer(strategy='mean',fill_value=50)
print(imputer.fit_transform(job))

### Median Imputation
imputer=SimpleImputer(strategy='median')
print(imputer.fit_transform(job))

################# Standard Scaling #######################

from sklearn.preprocessing import StandardScaler


milk =pd.read_csv("milk.csv",index_col=0)
print(milk)
scaler=StandardScaler()

scaler.fit(milk)
print(scaler.mean_)
print(scaler.scale_)

m_scl=scaler.transform(milk)
df_milk=pd.DataFrame(m_scl,columns=milk.columns,index=milk.index)
print(df_milk)

############Min Max Scaler###############
from sklearn.preprocessing import MinMaxScaler

milk =pd.read_csv("milk.csv",index_col=0)
print(milk)
mm=MinMaxScaler()

mm.fit(milk)
print(mm.min_)
print(mm.scale_)

mm_scl=mm.fit_transform(milk)
df_milk=pd.DataFrame(mm_scl,columns=milk.columns,index=milk.index)
print(df_milk)
