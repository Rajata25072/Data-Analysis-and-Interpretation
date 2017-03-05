# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 17:41:16 2016

@author: DIMEX
"""

import pandas
import numpy
import scipy.stats
import seaborn
import matplotlib.pyplot as plt

#import data
data = pandas.read_csv('ool_pds.csv', sep=',', low_memory=False)

#setting variables to be working with to numeric and category
data['PPAGECT4'] = data['PPAGECT4'].astype('category') #factor variable - age category
data['PPETHM'] = data['PPETHM'].astype('category') #factor variable - ethnic
data['W1_D9'] = data['W1_D9'].convert_objects(convert_numeric=True) #factor variable 
#- [Hillary Clinton] How would you rate
data['W1_P2'] = data['W1_P2'].astype('category') #factor variable - social class belong to
data['W1_F1'] = data['W1_F1'].astype('category') #factor variable 
#Future outlook - generally optimistic, pessimistic, or neither
data['W1_G3A'] = data['W1_G3A'].astype('category') #factor variable 
#economic outlook - generally optimistic, pessimistic, or neither

#######################################################################################################
#1.Make and implement data management decisions 
#No secondary variable
#1.1 Subset data
#1.1.1 Subset for selected variables
# subset variables in new data frame, sub1
sub1=data[['PPAGECT4','PPETHM', 'W1_D9', 'W1_P2', 'W1_F1', 'W1_G3A']]

#1.1.2 Subset for young-middle-age (18-44) - PPAGECT4 of 1-2
sub1=sub1[(data['PPAGECT4']==1) | (data['PPAGECT4']==2)]
sub2=sub1.copy()

#1.1.3 Coding out missing data (-1 = missing |
# No missing value for PPAGECT4 & PETHM)
sub2['W1_P2']=sub2['W1_P2'].replace(-1, numpy.nan) 
sub2['W1_F1']=sub2['W1_F1'].replace(-1, numpy.nan) 
sub2['W1_D9']=sub2['W1_D9'].replace(-1, numpy.nan) 
sub2['W1_D9']=sub2['W1_D9'].replace(998, numpy.nan)
sub2['W1_G3A']=sub2['W1_G3A'].replace(-1, numpy.nan)

sub2['W1_P2'] = sub2['W1_P2'].cat.remove_categories([-1]) #remove unused factor variable
sub2['W1_G3A'] = sub2['W1_G3A'].cat.remove_categories([-1]) #remove unused factor variable

ct1 = sub2.groupby('W1_G3A').size()
print (ct1)


#1.1.4 Recoding variable
#recoding values for W1_F1 into a new variable, W1_F1n
#to be more intuitive (-1=negative,0=neither,1=positive)
recode1 = {1: 1, 2: 0, 3: -1}
sub2['W1_F1n']= sub2['W1_F1'].map(recode1)
type (sub2['W1_F1n'])

#1.1.5 recoding ethnicity to be more intuituve
#White=1/Black=2/Other=3/Hispanic=4/2+=5
recode1 = {1: '1_White', 2: '2_Black', 3: '4_Other', 4: '3_Hispanic', 5: '5_2+Ethnic'}
sub2['PPETHM']= sub1['PPETHM'].map(recode1)

#1.1.6 Create secondary variable
#1) POSITIVE outlook
def POSITIVE (row):
   if row['W1_F1n'] == 1 : 
      return 1 
   elif (row['W1_F1n'] == 0) | (row['W1_F1n'] == -1) :
      return 0
   else :
       return numpy.nan
sub2['POSITIVE'] = sub2.apply (lambda row: POSITIVE (row),axis=1)

#2) NEGATIVE outlook
def NEGATIVE (row):
   if row['W1_F1n'] == -1 : 
      return 1 
   elif (row['W1_F1n'] == 0) | (row['W1_F1n'] == 1) :
      return 0    
   else :
       return numpy.nan     
sub2['NEGATIVE'] = sub2.apply (lambda row: NEGATIVE (row),axis=1)

#3)HILLARY favorness (favor v unfavor)
def HILLARY (row):
   if row['W1_D9'] == 10 : 
      return 1
   elif row['W1_D9'] <= 9:
      return 0  
sub2['HILLARY'] = sub2.apply (lambda row: HILLARY (row),axis=1)

#4) POSITIVE Economy outlook
def POSITIVE_E (row):
   if row['W1_G3A'] == 1 : 
      return 1 
   elif (row['W1_G3A'] == 0) | (row['W1_G3A'] == -1) :
      return 0
   else :
       return numpy.nan
sub2['POSITIVE_E'] = sub2.apply (lambda row: POSITIVE_E (row),axis=1)

#5) NEGATIVE outlook
def NEGATIVE_E (row):
   if row['W1_G3A'] == -1 : 
      return 1 
   elif (row['W1_G3A'] == 0) | (row['W1_G3A'] == 1) :
      return 0    
   else :
       return numpy.nan     
sub2['NEGATIVE_E'] = sub2.apply (lambda row: NEGATIVE_E (row),axis=1)



#######################################################################################################
#2. Perform chi-square test
#2.1 PPETHM v POSITIVE
#2.1.1 Contingency table of observed counts
ct1=pandas.crosstab(sub2['POSITIVE'], sub2['PPETHM'])
print (ct1)

#2.1.2 Column percentages
colsum=ct1.sum(axis=0)
colpct=ct1/colsum
print(colpct)

#2.1.3 Chi-square test
print ('chi-square value, p value, expected counts')
cs1= scipy.stats.chi2_contingency(ct1)
print (cs1)

#2.1.4 graph percent with POSITIVE outlook within each ethicity
seaborn.factorplot(x="PPETHM", y="POSITIVE", data=sub2, kind="bar", ci=None,  
                   order=["4_Other", "3_Hispanic", "5_2+Ethnic", "1_White", "2_Black"])
plt.xlabel('Ethinic Minority')
plt.ylabel('Percentage of Respondent Having Positive Outlook')


####################################
#2.2 PPETHM v NEGATIVE
#2.2.1 contingency table of observed counts
ct2 = pandas.crosstab(sub2['NEGATIVE'], sub2['PPETHM'])
print (ct2)

#2.2.2 Column percentages
colsum=ct2.sum(axis=0)
colpct=ct2/colsum
print(colpct)

#2.2.3 chi-square
print ('chi-square value, p value, expected counts')
cs2= scipy.stats.chi2_contingency(ct2)
print (cs2)

#2.2.4 graph percent with NEGATIVE outlook within each ethicity
seaborn.factorplot(x="PPETHM", y="NEGATIVE", data=sub2, kind="bar", ci=None,  
                   order=["3_Hispanic", "2_Black", "1_White", "4_Other", "5_2+Ethnic"])
plt.xlabel('Ethinic Minority')
plt.ylabel('Percentage of Respondent Having Negative Outlook')

#########################
#3. pair-wise comparisons for PPETHM v NEGATIVE
#3.1 3_Hispanic v 2_Black
recode2 = {'3_Hispanic': '3_Hispanic', '2_Black': '2_Black'}
sub2['COMP3v2']= sub2['PPETHM'].map(recode2)

# contingency table of observed counts
ct23v2=pandas.crosstab(sub2['NEGATIVE'], sub2['COMP3v2'])
print (ct23v2)

# column percentages
colsum=ct23v2.sum(axis=0)
colpct=ct23v2/colsum
print(colpct)

print ('chi-square value, p value, expected counts')
cs23v2= scipy.stats.chi2_contingency(ct23v2)
print (cs23v2)

####################################
#2.3 Social Class v NEGATIVE
#2.3.1 contingency table of observed counts
ct3 = pandas.crosstab(sub2['NEGATIVE'], sub2['W1_P2'])
print (ct3)

#2.3.2 column percentages
colsum=ct3.sum(axis=0)
colpct=ct3/colsum
print(colpct)

#2.3.3 chi-square
print ('chi-square value, p value, expected counts')
cs3= scipy.stats.chi2_contingency(ct3)
print (cs3)

#2.3.4 graph percent with with NEGATIVE outlook within each social class 
seaborn.factorplot(x="W1_P2", y="NEGATIVE", data=sub2, kind="bar", ci=None, order=[5, 4, 3, 1, 2])
plt.xlabel('Social Class')
plt.ylabel('Percentage of Respondent Having Negative Outlook')

####################################
#2.4 Social Class v HILLARY
#2.4.1 contingency table of observed counts
ct4 = pandas.crosstab(sub2['HILLARY'], sub2['W1_P2'])
print (ct4)

#2.2.2 Column percentages
colsum=ct4.sum(axis=0)
colpct=ct4/colsum
print(colpct)

#2.2.3 chi-square
print ('chi-square value, p value, expected counts')
cs4= scipy.stats.chi2_contingency(ct4)
print (cs4)

#2.2.4 graph percent with NEGATIVE outlook within each ethicity
seaborn.factorplot(x="W1_P2", y="HILLARY", data=sub2, kind="bar", ci=None, order=[1, 5, 3, 2, 4])
plt.xlabel('Ethinic Minority')
plt.ylabel('Percentage of Respondent Having Negative Outlook')


####################################
#2.5 Social Class v NEGATIVE_E
#2.4.1 contingency table of observed counts
ct5 = pandas.crosstab(sub2['NEGATIVE_E'], sub2['W1_P2'])
print (ct5)

#2.2.2 Column percentages
colsum=ct5.sum(axis=0)
colpct=ct5/colsum
print(colpct)

#2.2.3 chi-square
print ('chi-square value, p value, expected counts')
cs5= scipy.stats.chi2_contingency(ct4)
print (cs5)

#2.2.4 graph percent with NEGATIVE outlook within each ethicity
seaborn.factorplot(x="W1_P2", y="NEGATIVE_E", data=sub2, kind="bar", ci=None)
plt.xlabel('Ethinic Minority')
plt.ylabel('Percentage of Respondent Having Negative Outlook')