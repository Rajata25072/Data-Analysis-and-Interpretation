# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 18:24:45 2016

@author: DIMEX
"""
#######################################################
# 1. import library & data
#######################################################
# 1.1 import library 
import pandas
import statsmodels.formula.api as smf
import seaborn
import matplotlib.pyplot as plt

# bug fix for display formats to avoid run time errors
#pandas.set_option('display.float_format', lambda x:'%.2f'%x)

# 1.2 import data
data = pandas.read_csv('ool_pds.csv', sep=',', low_memory=False)

# 1.3 setting variables to be working with to numeric and category
data['W1_P2'] = pandas.to_numeric(data['W1_P2'], errors='coerce') 
#factor -> quantitative variable - social class belong to
# 1 Poor - 2 Working class - 3 Middle class - 4 Upper-middle class - 5 Upper class 
data['W1_F6'] = pandas.to_numeric(data['W1_F6'], errors='coerce') 
#quantitative variable - life outlook (quantitative)
#How far along the road to your American Dream do you think you will ultimately get
#on a 10-point scale where 1 is not far at all and 10 nearly there?

#######################################################
# 2. Make and implement data management decisions 
#######################################################
#2.1 Subset for selected variables
# subset variables in new data frame, sub1
sub1=data[['W1_P2', 'W1_F6']]

#2.2 Subset out the missing value for W1_P2
sub2=sub1[(sub1['W1_P2'] > -1)]
print (sub2["W1_P2"].value_counts(sort=False))

#2.3 Recode social class to make 0 for middle class and below & 1 for higher than middle class
recode1 = {1: 0, 2: 0, 3:0, 4:1, 5:1}
sub2['W1_P2']= sub2['W1_P2'].map(recode1)
print (sub2["W1_P2"].value_counts(sort=False))

############################################################################################
# 3. BASIC LINEAR REGRESSION
############################################################################################
# 3.1 Print scatterplot
#scat1 = seaborn.regplot(x="W1_P2", y="W1_F6", scatter=True, data=sub2)
plot1 = seaborn.factorplot(x="W1_P2", y="W1_F6", data=sub2, kind="bar", ci=None)
plt.xlabel('Social Class (0: Middle and Below vs 1: Higher than Middle)')
plt.ylabel('Mean of Life Outlook Score')
plt.title ('Bivariate Bar Graph for the Social Class and Life Outlook Score')
print(plot1)

# 3.2 Perform Regression Analysis
print ("OLS Regression Model for the Association Between Social Class and Life Outlook Score")
reg1 = smf.ols('W1_F6 ~ W1_P2', data=sub2).fit()
print (reg1.summary())