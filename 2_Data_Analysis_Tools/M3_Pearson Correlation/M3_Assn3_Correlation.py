# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 08:50:20 2016

@author: DIMEX
"""
####################################################################################
#0. Import library, data and variable conversion
#0.1import library
import pandas
import numpy
import seaborn
import scipy
import matplotlib.pyplot as plt

#0.2 import data
data = pandas.read_csv('ool_pds.csv', sep=',', low_memory=False)

#0.3 setting variables to be working with to numeric and category
data['W1_D1'] = data['W1_D1'].convert_objects(convert_numeric=True) 
#numeric variable - [Barack Obama] How would you rate
data['W1_D9'] = data['W1_D9'].convert_objects(convert_numeric=True) 
#numeric variable - [Hillary Clinton] How would you rate
data['W1_P2'] = data['W1_P2'].convert_objects(convert_numeric=True) 
#factor variable - social class belong to
data['W1_F6'] = data['W1_F6'].convert_objects(convert_numeric=True)
#factor variable - life outlook (quantitative)
#How far along the road to your American Dream do you think you will ultimately get
#on a 10-point scale where 1 is not far at all and 10 nearly there?

####################################################################################
#1.Make and implement data management decisions 
#No secondary variable
#1.1 Subset data
#1.1.1 Subset for selected variables
# subset variables in new data frame, sub1
sub1=data[['PPAGECT4','W1_D1','W1_D9','W1_P2','W1_F6']]

#1.1.2 Subset for young-middle-age (18-44) - PPAGECT4 of 1-2
sub1=sub1[(data['PPAGECT4']==1) | (data['PPAGECT4']==2)]
sub2=sub1.copy()

#1.1.3 Coding out missing data (-1 = missing |
# No missing value for PPAGECT4 & PETHM)
sub2['W1_D1']=sub2['W1_D1'].replace(-1, numpy.nan) 
sub2['W1_D1']=sub2['W1_D1'].replace(998, numpy.nan) 
sub2['W1_D9']=sub2['W1_D9'].replace(-1, numpy.nan) 
sub2['W1_D9']=sub2['W1_D9'].replace(998, numpy.nan)
sub2['W1_P2']=sub2['W1_P2'].replace(-1, numpy.nan) 
sub2['W1_F6']=sub2['W1_F6'].replace(-1, numpy.nan) 

####################################################################################
#2. Data Exploration
print ("W1_D1: Obama Score")
print (sub2["W1_D1"].value_counts(sort=False))

print ("W1_D9: Hillary Score")
print (sub2["W1_D9"].value_counts(sort=False))

print ("W1_P2: Social Class")
print (sub2["W1_P2"].value_counts(sort=False))

print ("W1_F6: Life Outlook")
print (sub2["W1_F6"].value_counts(sort=False))

####################################################################################
#3. Scatter Plot & correlation
sub_clean = sub2.dropna() #drop na to calculate correlation

#3.1 Obama vs Hilary score
#3.1.1 Scatterplot
scat1 = seaborn.regplot(x="W1_D1", y="W1_D9", data=sub_clean)
plt.xlabel('Rating Score for Barack Obama')
plt.ylabel('Rating Score for Hillary Clinton')
plt.title('Scatterplot between Obama and Hillary Rating Score') #somewhat positive relationship

#3.1.2 Correlation
print ('association between Obama and Hillary Rating Score')
print (scipy.stats.pearsonr(sub_clean['W1_D1'], sub_clean['W1_D9'])) 
#(0.65343666087794294, 2.1192342369025194e-96) -> significantly positive

#3.2 Obama score vs Life Outlook
#3.2.1 Scatterplot
scat2 = seaborn.regplot(x="W1_D1", y="W1_F6", data=sub_clean)
plt.xlabel('Rating Score for Barack Obama')
plt.ylabel('Rating Score for Life Outllok')
plt.title('Scatterplot between Obama Score and Life Outlook') #no clear relationship

#3.2.2 Correlation
print ('association between Obama Score and Life Outlook')
print (scipy.stats.pearsonr(sub_clean['W1_D1'], sub_clean['W1_F6'])) 
#(0.036008418121527955, 0.31457710144569967) -> insignificant

#3.3 Obama score vs social class
#3.3.1 Scatterplot
scat3 = seaborn.regplot(x="W1_P2", y="W1_D1", data=sub_clean)
plt.xlabel('Social Class')
plt.ylabel('Rating Score for Barack Obama')
plt.title('Scatterplot between Social Class and Obama Score') #somewhat negative relationship

#3.3.2 Correlation
print ('association between Social Class and Obama Scor')
print (scipy.stats.pearsonr(sub_clean['W1_P2'], sub_clean['W1_D1'])) 
#(-0.097671704238115775, 0.0062667745897747804) -> significantly negative
####################################################################################