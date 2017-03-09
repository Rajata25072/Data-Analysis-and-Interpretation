# -*- coding: utf-8 -*-
"""
The Decision Tree Assignment which aims to predict probability of citizen
having negative life outlook based on their ethnicity, social status and income.

@author: smallpi
"""

#######################################################
# 1. import library & data
#######################################################
# 1.1 import library 

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics

# 1.2 set directory
os.chdir("D:\!MOOC\Python_Directory\Data Analysis and Interpretation")

# 1.3 import data
data = pd.read_csv('ool_pds.csv', sep=',', low_memory=False)
data.dtypes
data.describe()

# 1.4 setting variables to be working with to numeric and category
# 1.4.1 Dependent variable
data['ool'] = data['W1_F1'].astype('category') #factor variable 

# 1.4.2 Explanatory variables
data['inc'] = pd.to_numeric(data['PPINCIMP'], errors='coerce') # qt var. - household income
data['soc'] = pd.to_numeric(data['W1_P2'], errors='coerce') # ft var. (can order) - social class 
data['ethm'] = data['PPETHM'].astype('category')  # ft var. (can't order) - ethnic
data['sex'] = data['PPGENDER'].astype('category') # ft var. (binary) - gender
data['edu'] = pd.to_numeric(data['PPEDUCAT'], errors='coerce') # ft var. (can order) - education
data['age'] = pd.to_numeric(data['PPAGE'], errors='coerce') # qt var. - age
data['unemp'] = data['W1_P11'].astype('category') # ft var. (binary) - unemploy


#######################################################
# 2. Make and implement data management decisions 
#######################################################
# 2.1 Subset for selected variables
# subset variables in new data frame, sub1
s_data=data[['ool', 'inc', 'soc', 'ethm','sex','edu','age','unemp']]
s_data.dtypes
s_data.describe() #describe quantitative var.

# 2.2 Coding out missing data (-1 = missing)
# No missing value for inc & ethm & sex & edu & age
s_data['ool'] = s_data['ool'].replace(-1, np.nan) 
s_data['soc'] = s_data['soc'].replace(-1, np.nan) 
s_data['unemp'] = s_data['unemp'].replace(-1, np.nan) 

data_clean = s_data.dropna()

data_clean.dtypes
data_clean.describe()

# 2.3 Recode variables
# 2.3.1 Recoding values for ool to be more intuitive
recode1 = {1: 'positive', 2: 'neutral', 3: 'negative'}
print (data_clean["ool"].value_counts(sort=False)) #before recoding
data_clean['ool']= data_clean['ool'].map(recode1)
print (data_clean["ool"].value_counts(sort=False)) #after recoding

# 2.3.2 Recode inc to be quantitative
recode1 = {1:2500, 2:6250, 3:8750, 4:11250, 5:13750, 6: 17500, 7:22500, 8: 27500, 9: 32500, 
           10:37500, 11: 45000, 12:55000, 13:67500, 14: 80000, 15: 92500, 16:112500, 
           17: 137500, 18: 162500, 19: 200000}
print (data_clean["inc"].value_counts(sort=False)) #before recoding
data_clean['inc']= data_clean['inc'].map(recode1)
print (data_clean["inc"].value_counts(sort=False)) #after recoding

# 2.3.3 Recode soc to start with 0
print (data_clean["soc"].value_counts(sort=False)) #before recoding
data_clean["soc"] = data_clean["soc"] -1
print (data_clean["soc"].value_counts(sort=False)) #after recoding

# 2.3.4 Recode sex to have 0 = male | 1 = female
recode1 = {1: 0, 2: 1}
data_clean['sex']= data_clean['sex'].map(recode1)
data_clean['sex'] = data_clean['sex'].astype('category') 
print (data_clean["sex"].value_counts(sort=False))

# 2.3.5 Recode unemp to have 0 = employ | 1 = unemploy
recode1 = {1: 1, 2: 0}
data_clean['unemp']= data_clean['unemp'].map(recode1)
data_clean['unemp'] = data_clean['unemp'].astype('category')
print (data_clean["unemp"].value_counts(sort=False))

# 2.4 Check the data
print (data_clean.dtypes)
print (data_clean.describe())

# 2.5 Set prediction and target variable
predictors = data_clean[['inc', 'soc', 'ethm','sex','edu','age','unemp']]
targets = data_clean['ool']

# 2.6 Split into training and testing sets
pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, targets, 
                                                                 test_size=.4)

pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape


##############################################################################
# 3. Perform Analysis
##############################################################################
# 3.1 Base Model
# 3.1.1 Create base model
print ('Training Set Frequency Table')
print (tar_train.value_counts(sort=False, normalize=True)) # -> always predict positive

# 3.1.2 Base model accuracy -> always predict positive
print ('Test Set Frequency Table')
print (tar_test.value_counts(sort=False, normalize=True)) # 0.55 accuracy

# 3.2 Decision Tree Model
# 3.2.1 Build model on training data
classifier=DecisionTreeClassifier(max_leaf_nodes = 5)
classifier=classifier.fit(pred_train,tar_train)
#help (DecisionTreeClassifier())

# 3.2.3 Checking the result - training set
prediction_train=classifier.predict(pred_train)
print ("Decision Tree - Training Set Result: Confusion Matrix & Accuracy")
print (sklearn.metrics.confusion_matrix(tar_train,prediction_train))
print (sklearn.metrics.accuracy_score(tar_train, prediction_train)) #model accuracy ~ 0.58

# 3.2.4 Checking the result - test set
prediction_test=classifier.predict(pred_test)
print ("Decision Tree - Test Set Result: Confusion Matrix & Accuracy")
print (sklearn.metrics.confusion_matrix(tar_test,prediction_test))
print (sklearn.metrics.accuracy_score(tar_test, prediction_test)) #model accuracy = 0.56

# 3.2.4 Displaying the decision tree
from sklearn import tree
#from StringIO import StringIO
from io import StringIO
from IPython.display import Image
out = StringIO()
tree.export_graphviz(classifier, out_file=out,
                     feature_names=pred_train.columns.values, 
                     class_names = ['negative', 'neutral', 'positive'],filled=True, rounded=True)
                     
import pydotplus
graph=pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png())