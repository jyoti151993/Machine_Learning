# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 18:19:35 2023

@author: dbda-lab
"""

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")


fp_df=pd.read_csv("Faceplate.csv", index_col=0)
fp_df.astype(bool)
itemsets=apriori(fp_df, min_support=0.2, use_colnames=True)



## Forming the rule

rules=association_rules(itemsets, metric='confidence', min_threshold=0.6)

type(rules)
print(rules.columns)

#### sort the rules as per lift metrics

rules=rules.sort_values(by='lift', ascending=False)


################### Cosmetic Datasets ########################
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Association")

cos_df=pd.read_csv("Cosmetics.csv", index_col=0)
cos_df.astype(bool)
itemsets=apriori(cos_df, min_support=0.05, use_colnames=True)

rules=association_rules(itemsets, metric='confidence', min_threshold=0.7)

type(rules)
print(rules.columns)
rules=rules.sort_values(by='lift', ascending=False)


############### catalog 

cat_df=pd.read_csv("CatalogCrossSell.csv", index_col=0)

cat_df.astype(bool)
itemsets=apriori(cat_df, min_support=0.05, use_colnames=True)


rules=association_rules(itemsets, metric='confidence', min_threshold=0.7)

type(rules)
print(rules.columns)
rules=rules.sort_values(by='lift', ascending=False)


############## Groceries Data processing  ##############
from mlxtend.preprocessing import TransactionEncoder

groceries=[]
with open("groceries.csv","r") as f:groceries =f.read()
groceries=groceries.split("\n")


groceries_list=[]
for i in groceries:
  groceries_list.append(i.split(","))

print(groceries_list)


te=TransactionEncoder()
te_ary=te.fit(groceries_list).transform(groceries_list)
te_ary


fp_df=pd.DataFrame(te_ary, columns=te.columns_)


itemsets=apriori(fp_df, min_support=0.005, use_colnames=True)


rules=association_rules(itemsets, metric='confidence', min_threshold=0.6)

type(rules)
print(rules.columns)
rules=rules.sort_values(by='lift', ascending=False)

############ DataSetA #############################
from mlxtend.preprocessing import TransactionEncoder
 
from mlxtend.

datasetA=[]
with open("DataSetA.csv","r") as f:datasetA =f.read()
datasetA=datasetA.split("\n")


Data_list=[]
for i in datasetA:
    Data_list.append(i.split(","))

print(Data_list)


te=TransactionEncoder()
te_ary=te.fit(Data_list).transform(Data_list)
te_ary


fp_df=pd.DataFrame(te_ary, columns=te.columns_)
fp_df=fp_df.iloc[:,1:]

itemsets=apriori(fp_df, min_support=0.005, use_colnames=True)


rules=association_rules(itemsets, metric='confidence', min_threshold=0.5)

type(rules)
print(rules.columns)
rules=rules.sort_values(by='lift', ascending=False)
