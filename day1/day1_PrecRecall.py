# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
os.chdir(r"C:\Users\dbda-lab\Desktop\ML\Datasets")
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


eval_data =pd.read_csv("eval_data.csv")
print(eval_data)
print("\n")

print(confusion_matrix(eval_data['y_test'],eval_data['y_pred']))


### ACCURACY
print("\n")
print(accuracy_score(eval_data['y_test'],eval_data['y_pred']))
print("\n")
## RECALL_SCORE
## recall(0)
print(recall_score(eval_data['y_test'],eval_data['y_pred'], pos_label=0))
print("\n")
## recall(1)
print(recall_score(eval_data['y_test'],eval_data['y_pred'], pos_label=1))
print("\n")

## Recall Average
print("Average_Recall",recall_score(eval_data['y_test'],eval_data['y_pred'],average='macro'))
print("\n")

##Recal Weighted Average
print("Recall Weighted Average",recall_score(eval_data['y_test'],eval_data['y_pred'],average='weighted'))

print("\n")



### PRECISION_SCORE

print(precision_score(eval_data['y_test'],eval_data['y_pred'], pos_label=0))
print("\n")
print(precision_score(eval_data['y_test'],eval_data['y_pred'], pos_label=1))

print("\n")

## Precision Average
print("Average_Precision",precision_score(eval_data['y_test'],eval_data['y_pred'],average='macro'))

print("\n")

## Precision  Weighted Average
print("Average_Precision",precision_score(eval_data['y_test'],eval_data['y_pred'],average='weighted'))



## F1_(0)
print("f1 Score for (0)",f1_score(eval_data['y_test'], eval_data['y_pred'], pos_label=0))

print("\n")
## F1_score(1)
print("f1 Score for (0)", f1_score(eval_data['y_test'], eval_data['y_pred'], pos_label=1))

print("\n")


# F1  score Average 

print("Average_F1",f1_score(eval_data['y_test'],eval_data['y_pred'],average='macro'))
      

print("\n")    
# F1 score weighted average    
print("WeightedAverage_F1",f1_score(eval_data['y_test'],eval_data['y_pred'],average='weighted'))



print("\n")

print(classification_report(eval_data['y_test'],eval_data['y_pred']))