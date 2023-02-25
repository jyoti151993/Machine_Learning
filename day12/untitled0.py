# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 09:34:49 2023

@author: dbda-lab
"""

from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np # linear algebra
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")
df= pd.read_csv("monthly-milk-production-pounds-p.csv")

df.plot()
plt.show()
series=df["Milk"]
result=seasonal_decompose(series,model='additive',period=12)
result.plot()
plt.show()

#########mulyiplicative decomposition########

series=df["Milk"]
result=seasonal_decompose(series,model='multiplicative',period=12)
result.plot()
plt.show()
### import smoothing#########
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing, Holt

y=df['Milk']
y_train=y[:-12]
y_test=y[-12:]

y_train.shape,y_test.shape,df.shape

##############gasoline dataset##########


gasoline= pd.read_csv("Gasoline.csv")
sales=gasoline['Sales']
fcast= sales.rolling(3,center=True).mean()
plt.plot(sales, label='Original Data')
plt.plot(fcast, label='Moving Average Forecast')

plt.legend(loc='best')
plt.show
df.plot()
plt.show()
####################### milk dataset#######################


df= pd.read_csv("monthly-milk-production-pounds-p.csv")
milk= df['Milk']

fcast=milk.rolling(3, center=True).mean()

plt.plot(milk, label='Original Data')
plt.plot(fcast, label='Moving Average Forecast')
plt.legend(loc='best')
plt.show()
# This is centered MA
span=2
#Trailing MA
fcast=y_train.rolling(span).mean()
MA=fcast.iloc[-1]
MA_series=pd.Series(MA.repeat(len(y_test)))
MA_fcast=pd.concat([fcast,MA_series],ignore_index=True)
plt.plot(y_train,label='Train')
plt.plot(y_test,label='Test')
plt.plot(MA_fcast,label='Moving Average Forecast')
plt.legend(loc='best')
plt.show()
print(np.sqrt(mean_squared_error(y_test,MA_series)))


span=3
#Trailing MA
fcast=y_train.rolling(span).mean()
MA=fcast.iloc[-1]
MA_series=pd.Series(MA.repeat(len(y_test)))
MA_fcast=pd.concat([fcast,MA_series],ignore_index=True)
plt.plot(y_train,label='Train')
plt.plot(y_test,label='Test')
plt.plot(MA_fcast,label='Moving Average Forecast')
plt.legend(loc='best')
plt.show()
print(np.sqrt(mean_squared_error(y_test,MA_series)))

span=4
#Trailing MA
fcast=y_train.rolling(span).mean()
MA=fcast.iloc[-1]
MA_series=pd.Series(MA.repeat(len(y_test)))
MA_fcast=pd.concat([fcast,MA_series],ignore_index=True)
plt.plot(y_train,label='Train')
plt.plot(y_test,label='Test')
plt.plot(MA_fcast,label='Moving Average Forecast')
plt.legend(loc='best')
plt.show()
print(np.sqrt(mean_squared_error(y_test,MA_series)))


span=5
#Trailing MA
fcast=y_train.rolling(span).mean()
MA=fcast.iloc[-1]
MA_series=pd.Series(MA.repeat(len(y_test)))
MA_fcast=pd.concat([fcast,MA_series],ignore_index=True)
plt.plot(y_train,label='Train')
plt.plot(y_test,label='Test')
plt.plot(MA_fcast,label='Moving Average Forecast')
#plt.plot(Milk, label='Original Data')
plt.legend(loc='best')
plt.show()
print(np.sqrt(mean_squared_error(y_test,MA_series)))