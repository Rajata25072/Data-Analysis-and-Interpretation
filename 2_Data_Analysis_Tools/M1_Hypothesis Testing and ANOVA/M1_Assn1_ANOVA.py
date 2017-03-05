# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 03:13:22 2016

@author: DIMEX
"""

import library
import pandas
import numpy
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi 
import seaborn
import matplotlib.pyplot as plt

#import data
data = pandas.read_csv('ool_pds.csv', sep=',', low_memory=False)

#setting variables to be working with to numeric and category
data['PPETHM'] = data['PPETHM'].astype('category') #factor variable - ethnic
data['W1_D1'] = data['W1_D1'].convert_objects(convert_numeric=True) #factor variable 
#- [Barack Obama] How would you rate
data['W1_D9'] = data['W1_D9'].convert_objects(convert_numeric=True) #factor variable 
#- [Hillary Clinton] How would you rate
data['W1_F6'] = data['W1_F6'].convert_objects(convert_numeric=True) #factor variable - life outlook (quantitative)
#How far along the road to your American Dream do you think you will ultimately get
#on a 10-point scale where 1 is not far at all and 10 nearly there?

#1. Make and implement data management decisions 
#No secondary variable
#1.1 Subset data
#1.1.1 Subset for selected variables
# subset variables in new data frame, sub1
sub1=data[['PPETHM', 'W1_D1','W1_D9', 'W1_F6', 'PPAGECT4']]

#1.1.2 Subset for young-middle-age (18-44) - PPAGECT4 of 1-2
sub1=sub1[(data['PPAGECT4']==1) | (data['PPAGECT4']==2)]
sub2=sub1.copy()

#1.1.3 Coding out missing data (-1 & 998= missing)
# No missing value for PETHM
sub2['W1_D1']=sub2['W1_D1'].replace(-1, numpy.nan) 
sub2['W1_D1']=sub2['W1_D1'].replace(998, numpy.nan) 
sub2['W1_D9']=sub2['W1_D9'].replace(-1, numpy.nan) 
sub2['W1_D9']=sub2['W1_D9'].replace(998, numpy.nan) 
sub2['W1_F6']=sub2['W1_F6'].replace(-1, numpy.nan) 

#1.1.4 Check after code out missing data
print (sub2['W1_D1'].value_counts(sort=False, dropna=False).sort_index())
print (sub2['W1_D9'].value_counts(sort=False, dropna=False).sort_index())
print (sub2['W1_F6'].value_counts(sort=False, dropna=False).sort_index())

#1.1.5 recoding ethnicity to be more intuituve
#White=1/Black=2/Other=3/Hispanic=4/2+=5
recode1 = {1: '1_White', 2: '2_Black', 3: '4_Other', 4: '3_Hispanic', 5: '5_2+Ethnic'}
sub2['PPETHM']= sub1['PPETHM'].map(recode1)

#1.1.5 Check size for each ethnic group
ct1 = sub2.groupby('PPETHM').size()
print (ct1)

#2. Performing ANOVA Analysis
#2.1 Using ols function for calculating the F-statistic and associated p value
print ('-------OLS Results for Hilary Score Using ethinicty as factor variable-------')
model1 = smf.ols(formula='W1_D9 ~ C(PPETHM)', data=sub2)
results1 = model1.fit()
print (results1.summary())

#2.2 Calculate mean & sd for score from each ethnicity
sub3 = sub2[['W1_D9', 'PPETHM']].dropna()

print ('means for W1_D9 by ethnicity')
m1= sub3.groupby('PPETHM').mean()
print (m1)

print ('standard deviations for W1_D9 by ethnicity')
sd1 = sub3.groupby('PPETHM').std()
print (sd1)

#2.3 Perform ad-hoc test using tukey
mc1 = multi.MultiComparison(sub3['W1_D9'], sub3['PPETHM'])
res1 = mc1.tukeyhsd()
print ('------Ad-hoc Test Sesults for Comparing Score Among Ethnicity---------')
print(res1.summary())
#Black ethnicity is significantly higher score from other major groups,
#namely, Mixed-race, White and Hispanic (The other race has mixed result)

#2.4 Further bar plot
seaborn.factorplot(x='PPETHM', y='W1_D9', data=m1, kind="bar", ci=None)
plt.xlabel('Hilary Rating')
plt.ylabel('Ethinicity')