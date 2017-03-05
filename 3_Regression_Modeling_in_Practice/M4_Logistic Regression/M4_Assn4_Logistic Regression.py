# -*- coding: utf-8 -*-
"""
The Logistic Regression Assignment which aims to predict probability of citizen
having negative life outlook based on their ethnicity, social status and income.

@author: smallpi
"""

#######################################################
# 1. import library & data
#######################################################
# 1.1 import library 
import numpy
import pandas
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn

# bug fix for display formats to avoid run time errors
#pandas.set_option('display.float_format', lambda x:'%.2f'%x)

# 1.2 import data
data = pandas.read_csv('ool_pds.csv', sep=',', low_memory=False)

# 1.3 setting variables to be working with to numeric and category
# 1.3.1 Dependent variable
data['outlook_qt'] = data['W1_F6'].convert_objects(convert_numeric=True) 
# dependent variable - life outlook (quantitative)
#How far along the road to your American Dream do you think you will ultimately get
#on a 10-point scale where 1 is not far at all and 10 nearly there?
data['outlook_ql'] = data['W1_F1'].astype('category') #factor variable 
#Future outlook - generally optimistic, pessimistic, or neither

# 1.3.2 Explanatory variables
data['hh_income'] = pandas.to_numeric(data['PPINCIMP'], errors='coerce')
# quantitative variable - household income
data['soc_class'] = pandas.to_numeric(data['W1_P2'], errors='coerce')
# factor variable (can be ordered) - social class belong to
data['ethnicity'] = data['PPETHM'].astype('category') 
# factor variable (can't be ordered) - ethnic


#######################################################
# 2. Make and implement data management decisions 
#######################################################
# 2.1 Subset for selected variables
# subset variables in new data frame, sub1
sub1=data[['outlook_ql', 'hh_income', 'soc_class', 'ethnicity']]

# 2.2 Coding out missing data (-1 & 998 = missing)
# No missing value for hh_income & ethnicity
sub1['outlook_ql']=sub1['outlook_ql'].replace(-1, numpy.nan) 
sub1['soc_class']=sub1['soc_class'].replace(-1, numpy.nan) 

# 2.3 Recode variables
# 2.3.1 Recoding values for outlook_ql into a new variable, W1_F1n
#to be more intuitive (-1=negative,0=neither,1=positive)
recode1 = {1: 1, 2: 0, 3: -1}
sub1['outlook_ql']= sub1['outlook_ql'].map(recode1)
type (sub1['outlook_ql'])

# 2.3.2 Recode household income to be quantitative
recode1 = {1:2500, 2:6250, 3:8750, 4:11250, 5:13750, 6: 17500, 7:22500, 8: 27500, 9: 32500, 
           10:37500, 11: 45000, 12:55000, 13:67500, 14: 80000, 15: 92500, 16:112500, 
           17: 137500, 18: 162500, 19: 200000}
print (sub1["hh_income"].value_counts(sort=False)) #before recoding
sub1['hh_income']= sub1['hh_income'].map(recode1)
print (sub1["hh_income"].value_counts(sort=False)) #after recoding

# 2.3.3 Recode social class so that the lowest social class starts with 0
print (sub1['soc_class'].value_counts(sort=False)) #before recoding
sub1['soc_class'] = sub1['soc_class'] - 1
print (sub1['soc_class'].value_counts(sort=False)) #after recoding

# 2.4 Create secondary variable: NEGATIVE outlook
def NEGATIVE (row):
   if row['outlook_ql'] == -1 : 
      return 1 
   elif (row['outlook_ql'] == 0) | (row['outlook_ql'] == 1) :
      return 0    
   else :
       return numpy.nan     
sub1['neg_outlook'] = sub1.apply (lambda row: NEGATIVE (row),axis=1)

# Check for correction
print (sub1["outlook_ql"].value_counts(sort=False)) 
print (sub1["neg_outlook"].value_counts(sort=False)) 

# 2.5 Further adjust data for regression analysis
# 2.5.1 center IV (household income)
sub1['hh_income_c'] = sub1['hh_income'] - sub1['hh_income'].mean()
sub1['hh_income_c'].describe()

# 2.5.2 hh_income has too small unit -> change unit from USD to 10,000 USD
sub1['hh_income_c'] = sub1['hh_income_c'] / 10000
sub1['hh_income_c'].describe()

#######################################################
# 3. Regression Analysis
#######################################################
# 4.1 with household income
reg1 = smf.ols('neg_outlook ~ hh_income_c', data=sub1).fit()
print (reg1.summary()) #Adj. R-squared = -0.000 #household income insignificant

# 4.2 Wiht social class
reg2 = smf.ols('neg_outlook ~ soc_class', data=sub1).fit()
print (reg2.summary()) #Adj. R-squared = 0.002

# 4.3 Adding ethnicity
reg3 = smf.ols('neg_outlook ~ soc_class + C(ethnicity)', data=sub1).fit()
print (reg3.summary()) #Adj. R-squared = 0.002 -> 0.045

# Best model = social class + ethnicity -> reg3


##############################################################################
# 4. Logistic Regression
##############################################################################
# 4.1 logistic regression with household income
lreg1 = smf.logit(formula = 'neg_outlook ~ hh_income_c', data = sub1).fit()
print (lreg1.summary()) #household income insignificant
# odds ratios
print ("Odds Ratios")
print (numpy.exp(lreg1.params))

# 4.2 logistic regression with household income and social class
lreg2 = smf.logit(formula = 'neg_outlook ~ soc_class', data = sub1).fit()
print (lreg2.summary())

# odd ratios with 95% confidence intervals
params = lreg2.params
conf = lreg2.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (numpy.exp(conf))

# 4.3 logistic regression with household income, social class and ethnicity
lreg3 = smf.logit(formula = 'neg_outlook ~ soc_class + C(ethnicity)', data = sub1).fit()
print (lreg3.summary())

# odd ratios with 95% confidence intervals
print ("Odds Ratios")
params = lreg3.params
conf = lreg3.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (numpy.exp(conf))