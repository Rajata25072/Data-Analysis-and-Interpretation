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
 # Feature Importance
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier

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
# 2.3.1 Recoding values for outlook_ql into a new variable, W1_F1n
#to be more intuitive (-1=negative,0=neither,1=positive)
recode1 = {1: 1, 2: 0, 3: -1}
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

# 2.4 Create secondary variable: POSITIVE outlook
def POSITIVE (row):
   if row['ool'] == 1 : 
      return 1 
   else :
       return 0
data_clean['pos'] = data_clean.apply (lambda row: POSITIVE (row),axis=1)
data_clean['pos'] = data_clean['pos'].astype('category') 

# 2.5 Check the data
data_clean.dtypes
data_clean.describe()

# 2.5 Set prediction and target variable
predictors = data_clean[['inc', 'soc', 'ethm','sex','edu','age','unemp']]
targets = data_clean['pos']

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
tar_train.describe() # 1 (positive) more often -> always predict positive
# 3.1.2 Base model accuracy -> always predict not negative
tar_test.describe() # 0.56

# 3.2 Random Forest 
# 3.2.1 Build model on training data
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=25)
classifier=classifier.fit(pred_train,tar_train)

# 3.2.2 Predict on test set
predictions=classifier.predict(pred_test)

# 3.2.3 Random Forest Accuracy
sklearn.metrics.confusion_matrix(tar_test,predictions)
sklearn.metrics.accuracy_score(tar_test, predictions) #0.57

# 3.2.4 Fit an Extra Trees model to the data to find attribute importance
model = ExtraTreesClassifier()
model.fit(pred_train,tar_train)
# display the relative importance of each attribute
var_name = (pred_train.columns.tolist())
var_sig = (list(model.feature_importances_))

# combine to 1 data frame
var_imp = DataFrame(columns=var_name)
var_imp.loc['Imp'] = [list(model.feature_importances_)[n] for n in range(7)]

# sort by importance
var_imp[var_imp.ix[var_imp.last_valid_index()].argsort()[::-1]]

"""
Running a different number of trees and see the effect
 of that on the accuracy of the prediction
"""
trees=range(25)
accuracy=np.zeros(25)

for idx in range(len(trees)):
   classifier=RandomForestClassifier(n_estimators=idx + 1)
   classifier=classifier.fit(pred_train,tar_train)
   predictions=classifier.predict(pred_test)
   accuracy[idx]=sklearn.metrics.accuracy_score(tar_test, predictions)
   
plt.cla()
plt.plot(trees, accuracy)
plt.ylabel('Accuracy')
plt.xlabel('Number of Trees')