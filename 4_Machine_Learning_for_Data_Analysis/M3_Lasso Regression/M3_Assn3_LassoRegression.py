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
#from pandas import Series, DataFrame
import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LassoLarsCV

# 1.2 set directory
os.chdir("D:\!MOOC\Python_Directory\Data Analysis and Interpretation")

# 1.3 import data
data = pd.read_csv('OOL_pds.csv', sep=',', low_memory=False)
data.columns = map(str.upper, data.columns) #upper-case all DataFrame column names
data.dtypes
data.describe()

# 1.4 setting variables to be working with to numeric and category
# 1.4.1 Dependent variable
data['OOL'] = data['W1_F6'].convert_objects(convert_numeric=True) 
# dependent variable - life outlook (quantitative)
#How far along the road to your American Dream do you think you will ultimately get
#on a 10-point scale where 1 is not far at all and 10 nearly there?

# 1.4.2 Explanatory variables -> must add more var
data['SEX'] = data['PPGENDER'].astype('category') # ft var. (binary) - gender
data['AGE'] = pd.to_numeric(data['PPAGE'], errors='coerce') # qt var. - age
data['EDU'] = pd.to_numeric(data['PPEDUCAT'], errors='coerce') # ft var. (can order) - education
data['INC'] = pd.to_numeric(data['PPINCIMP'], errors='coerce') # qt var. - household income
data['MARITAL'] = data['PPMARIT'].astype('category') #factor variable - marital status
data['ETHM'] = data['PPETHM'].astype('category')  # ft var. (can't order) - ethnic
data['SOC'] = pd.to_numeric(data['W1_P2'], errors='coerce') # ft var. (can order) - social class 
data['SOC_F'] = data['W1_P3'].convert_objects(convert_numeric=True) #factor variable - social class family belong to
data['UNEMP'] = data['W1_P11'].astype('category') # ft var. (binary) - unemploy
data['OBAMA'] = data['W1_D1'].convert_objects(convert_numeric=True) #rating of [Barack Obama] 
data['HILLARY'] = data['W1_D9'].convert_objects(convert_numeric=True) #rating of [Hillary Clinton] 


#######################################################
# 2. Make and implement data management decisions 
#######################################################
# 2.1 Subset for selected variables
# subset variables in new data frame, sub1
s_data=data[['OOL','SEX','AGE','EDU','INC','MARITAL','ETHM','SOC','SOC_F','UNEMP','OBAMA','HILLARY']]
s_data.dtypes
s_data.describe() #describe quantitative var.

# 2.2 Coding out missing data (-1 = missing)
# No missing value for SEX & AGE & EDU & INC & MARITAL & ETHM 
s_data['OOL'] = s_data['OOL'].replace(-1, np.nan) 
s_data['SOC'] = s_data['SOC'].replace(-1, np.nan) 
s_data['SOC_F'] = s_data['SOC_F'].replace(-1, np.nan) 
s_data['UNEMP'] = s_data['UNEMP'].replace(-1, np.nan) 
s_data['OBAMA'] = s_data['OBAMA'].replace(-1, np.nan) 
s_data['OBAMA'] = s_data['OBAMA'].replace(998, np.nan) 
s_data['HILLARY'] = s_data['HILLARY'].replace(-1, np.nan) 
s_data['HILLARY'] = s_data['HILLARY'].replace(998, np.nan) 

data_clean = s_data.dropna()
data_clean.dtypes
data_clean.describe()

# 2.3 Recode variables
# 2.3.1 Recode inc to be quantitative
recode1 = {1:2500, 2:6250, 3:8750, 4:11250, 5:13750, 6: 17500, 7:22500, 8: 27500, 9: 32500, 
           10:37500, 11: 45000, 12:55000, 13:67500, 14: 80000, 15: 92500, 16:112500, 
           17: 137500, 18: 162500, 19: 200000}
print (data_clean['INC'].value_counts(sort=False)) #before recoding
data_clean['INC']= data_clean['INC'].map(recode1)
print (data_clean['INC'].value_counts(sort=False)) #after recoding

# 2.3.3 Recode soc to start with 0
print (data_clean['SOC'].value_counts(sort=False)) #before recoding
data_clean['SOC'] = data_clean['SOC'] -1
print (data_clean['SOC'].value_counts(sort=False)) #after recoding

# 2.3.4 Recode sex to have 0 = male | 1 = female
recode1 = {1: 0, 2: 1}
data_clean['SEX']= data_clean['SEX'].map(recode1)
data_clean['SEX'] = data_clean['SEX'].astype('category') 
print (data_clean['SEX'].value_counts(sort=False))

# 2.3.5 Recode unemp to have 0 = employ | 1 = unemploy
recode1 = {1: 1, 2: 0}
data_clean['UNEMP']= data_clean['UNEMP'].map(recode1)
data_clean['UNEMP'] = data_clean['UNEMP'].astype('category')
print (data_clean['UNEMP'].value_counts(sort=False))

# 2.4 Check the data
data_clean.dtypes
data_clean.describe()

# 2.5 Set prediction and target variable
predvar = data_clean[['SEX','AGE','EDU','INC','MARITAL','ETHM','SOC','SOC_F','UNEMP','OBAMA','HILLARY']]
target = data_clean['OOL']

# 2.6 standardize predictors to have mean=0 and sd=1
predictors=predvar.copy()
from sklearn import preprocessing
predictors['SEX']=preprocessing.scale(predictors['SEX'].astype('float64'))
predictors['AGE']=preprocessing.scale(predictors['AGE'].astype('float64'))
predictors['EDU']=preprocessing.scale(predictors['EDU'].astype('float64'))
predictors['INC']=preprocessing.scale(predictors['INC'].astype('float64'))
predictors['MARITAL']=preprocessing.scale(predictors['MARITAL'].astype('float64'))
predictors['ETHM']=preprocessing.scale(predictors['ETHM'].astype('float64'))
predictors['SOC']=preprocessing.scale(predictors['SOC'].astype('float64'))
predictors['SOC_F']=preprocessing.scale(predictors['SOC_F'].astype('float64'))
predictors['UNEMP']=preprocessing.scale(predictors['UNEMP'].astype('float64'))
predictors['OBAMA']=preprocessing.scale(predictors['OBAMA'].astype('float64'))
predictors['HILLARY']=preprocessing.scale(predictors['HILLARY'].astype('float64'))

# 2.7 Check the data
predictors.dtypes
predictors.describe()

# 2.8 Split into training and testing sets
pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, target, 
                                                                 test_size=.3, random_state=123)

print (pred_train.shape)
print (pred_test.shape)
print (tar_train.shape)
print (tar_test.shape)


##############################################################################
# 3. Perform Analysis
##############################################################################
# 3.1 Specify the lasso regression model
model=LassoLarsCV(cv=10, precompute=False).fit(pred_train,tar_train)

# 3.2 print variable names and regression coefficients & sort by value
coef = dict(zip(predictors.columns, model.coef_))
import operator
sorted(coef.items(), key=operator.itemgetter(1), reverse=True) 
# most significants + are SOC, INC, EDU
# Non-significants are SEX, OBAMA, ETHM, HILLARY

# 3.3 Plot coefficient progression
# show the order of selected cofficient and its value when new predictors are added
m_log_alphas = -np.log10(model.alphas_) #alpha = penalty parameter = lambda through the model selection process
ax = plt.gca()
plt.plot(m_log_alphas, model.coef_path_.T) #.T = transpose
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')
# first 3 line = SOC, INC, EDU

# 3.4 Plot mean square error for each fold
# change in MSE for change in alpha
m_log_alphascv = -np.log10(model.cv_alphas_)
plt.figure()
plt.plot(m_log_alphascv, model.cv_mse_path_, ':')
plt.plot(m_log_alphascv, model.cv_mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')
# There is variability across individual cv as variables area added in the same pattern
# = Decrease rapidly and then level off to point where more prediction is not reducing MSE
         
# 3.5 MSE from training and test data
from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(tar_train, model.predict(pred_train))
test_error = mean_squared_error(tar_test, model.predict(pred_test))
print ('training data MSE')
print(train_error)
print ('test data MSE')
print(test_error) #similar accuracy

# 3.6 R-square from training and test data
rsquared_train=model.score(pred_train,tar_train)
rsquared_test=model.score(pred_test,tar_test)
print ('training data R-square')
print(rsquared_train)
print ('test data R-square')
print(rsquared_test) #more accurate than training data