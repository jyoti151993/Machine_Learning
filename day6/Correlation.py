# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 09:18:41 2023

@author: dbda-lab
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")

pizza=pd.read_csv('pizza.csv')

plt.scatter(pizza['Promote'],pizza['Sales'])
plt.show()
## hence we conclude/infer that this is high degree positivity 
# we can measure the correlation coeff
pizza['Promote'].corr(pizza['Sales']) # strong positive correlation


########## Insurance ##########
insurance=pd.read_csv("Insure_auto.csv", index_col=0)
plt.scatter(insurance['Operating_Cost'], insurance['Home'])
plt.show()
insurance['Operating_Cost'].corr(insurance['Home']) ## 0.945 is highest
insurance['Operating_Cost'].corr(insurance['Automobile']) #0.786
insurance['Home'].corr(insurance['Automobile']) ## 0.5488


### Correlation Matrix  
insurance.corr() # correlation of every var with evry other var also  called as correlation matrix

# Scatter Matrix plot
sns.pairplot(insurance)
plt.show()


#### HEATMAP -> always sort the data for multiple variables
sns.heatmap(data=insurance.corr(), annot=True) # annot is true we can see the values of correlation coefficients
plt.show()

###### boston
boston=pd.read_csv("Boston.csv")
plt.figure(figsize=(10,10))
sns.heatmap(boston.corr(), annot=True)
plt.show()
boston.corr()

