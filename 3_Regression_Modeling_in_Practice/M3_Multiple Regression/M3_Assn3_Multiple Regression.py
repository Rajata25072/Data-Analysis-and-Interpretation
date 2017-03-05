# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 12:28:47 2016

Assn3: Multiple Regression
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
data['lifeoutlook'] = data['W1_F6'].convert_objects(convert_numeric=True) 
# dependent variable - life outlook (quantitative)
#How far along the road to your American Dream do you think you will ultimately get
#on a 10-point scale where 1 is not far at all and 10 nearly there?

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
sub1=data[['lifeoutlook', 'hh_income', 'soc_class', 'ethnicity']]

# 2.2 Subset out the missing value for W1_P2
# -1 for lifeoutlook | no for hh_income | -1 for soc_class | no for ethnicity
sub2=sub1[(sub1['lifeoutlook'] > -1)]
print (sub1["lifeoutlook"].value_counts(sort=False)) #before coding out missing value 
print (sub2["lifeoutlook"].value_counts(sort=False)) #after coding out missing value 

sub3 = sub2[(sub2['soc_class'] > -1)]
print (sub2['soc_class'].value_counts(sort=False)) #before coding out missing value 
print (sub3['soc_class'].value_counts(sort=False)) #after coding out missing value

# 2.3 Subset for only black & white ethnicity
sub4 = sub3[((sub3['ethnicity'] == 1) | (sub3['ethnicity'] == 2))]
print (sub3['ethnicity'].value_counts(sort=False)) #before leaving out other ethnics
print (sub4['ethnicity'].value_counts(sort=False)) #after leaving out other ethnics

# 2.4 Recode variables
# 2.4.1 Recode household income to be quantitative
recode1 = {1:2500, 2:6250, 3:8750, 4:11250, 5:13750, 6: 17500, 7:22500, 8: 27500, 9: 32500, 
           10:37500, 11: 45000, 12:55000, 13:67500, 14: 80000, 15: 92500, 16:112500, 
           17: 137500, 18: 162500, 19: 200000}
print (sub4["hh_income"].value_counts(sort=False)) #before recoding
sub4['hh_income']= sub4['hh_income'].map(recode1)
print (sub4["hh_income"].value_counts(sort=False))

# 2.4.2 Recode social class so that the lowest social class starts with 0
print (sub4['soc_class'].value_counts(sort=False)) #before recoding
sub4['soc_class'] = sub4['soc_class'] - 1
print (sub4['soc_class'].value_counts(sort=False)) #after recoding

# 2.5 Further adjust data for regression analysis
# 2.5.1 center IV (household income)
sub4['hh_income_c'] = sub4['hh_income'] - sub4['hh_income'].mean()
sub4['hh_income_c'].describe()

# 2.5.2 hh_income has too small unit -> change unit from USD to 10,000 USD
sub4['hh_income_c'] = sub4['hh_income_c'] / 10000
sub4['hh_income_c'].describe()


#######################################################
# 3. Explore with scatterplot & barplot
#######################################################
# 3.1 bar plost ethnicity vs lifeotlook
seaborn.factorplot(x="ethnicity", y="lifeoutlook", data=sub4, kind="bar", ci=None)
plt.xlabel('Ethnic Minority')
plt.ylabel('Average Life Outlook Score') #not that different

# 3.2 bar plot soc_class vs lifeoutlook
seaborn.factorplot(x="soc_class", y="lifeoutlook", data=sub4, kind="bar", ci=None)
plt.xlabel('Social Class')
plt.ylabel('Average Life Outlook Score') #significant trend*

# 3.3 scatterplot for household income vs life outlook (first order)
scat1 = seaborn.regplot(x="hh_income", y="lifeoutlook", scatter=True, data=sub4)
plt.xlabel('Household Income')
plt.ylabel('Life Outlook Score') #significant trend*

# 3.4 scatterplot for household income vs life outlook (second order)
scat2 = seaborn.regplot(x="hh_income", y="lifeoutlook", scatter=True, order=2, data=sub4)
plt.xlabel('Household Income')
plt.ylabel('Life Outlook Score') #significant trend*

#######################################################
# 4. Regression Analysis
#######################################################
# 4.1 Linear Regression Analysis
reg1 = smf.ols('lifeoutlook ~ hh_income_c', data=sub4).fit()
print (reg1.summary())

# 4.2 Quadratic (polynomial) regression analysis
reg2 = smf.ols('lifeoutlook ~ hh_income_c + I(hh_income_c**2)', data=sub4).fit()
print (reg2.summary()) #Adj. R-squared change little (0.110 -> 0.116) -> quadratic insignificant

# 4.3 Adding social class
reg3 = smf.ols('lifeoutlook ~ hh_income_c + soc_class', data=sub4).fit()
print (reg3.summary()) #Adj. R-squared change significantly (0.110 -> 0.204)

# 4.4 Adding ethnicity
reg4 = smf.ols('lifeoutlook ~ hh_income_c + soc_class + ethnicity', data=sub4).fit()
print (reg4.summary()) #Adj. R-squared change little (0.204 -> 0.206) -> insignificant

# Best model = household_income + social class -> reg3
# intercept: ppl with average income and lowest social class has 5.07 score
# hh_income_c: for 10,000 more income, -> 0.07 more score
# soc_class: for 1 class higher -> 0.9 more score

#######################################################
# 5. Evaluating Model Fit
#######################################################
# 5.1 Q-Q plot for normality
fig1 = sm.qqplot(reg3.resid, line='r') 
# the closer to diagonal line, the more normality it has
# somewhat normal with exception on the very high end

# 5.2 simple plot of residuals
stdres=pandas.DataFrame(reg3.resid_pearson)
plt.plot(stdres, 'o', ls='None')
l = plt.axhline(y=2, color='r')
l = plt.axhline(y=0, color='r')
l = plt.axhline(y=-2, color='r')
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')
# close to 5% observation out of 2sd range

# 5.3 Additional regression diagnostic plots -> household income
fig2 = plt.figure(figsize=(8,6))
fig2 = sm.graphics.plot_regress_exog(reg3, "hh_income_c", fig=fig2)
print(fig2)
# residual fannel in for household income
# residual is somewhat randomly distributed?

# 5.4 Leverage plot
fig3 = sm.graphics.influence_plot(reg3, size=3)
l = plt.axhline(y=2, color='r')
l = plt.axhline(y=-2, color='r')
print(fig3)
#1 Outlier with quite high leverage