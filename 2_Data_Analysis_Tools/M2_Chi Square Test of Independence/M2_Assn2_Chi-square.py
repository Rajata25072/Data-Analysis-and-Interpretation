# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 21:17:19 2016

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
data['W1_F1'] = data['W1_F1'].astype('category') #factor variable 
#Future outlook - generally optimistic, pessimistic, or neither

#######################################################################################################
#1.Make and implement data management decisions 
#No secondary variable
#1.1 Subset data
#1.1.1 Subset for selected variables
# subset variables in new data frame, sub1
sub1=data[['PPAGECT4','PPETHM', 'W1_F1']]

#1.1.2 Subset for young-middle-age (18-44) - PPAGECT4 of 1-2
sub1=sub1[(data['PPAGECT4']==1) | (data['PPAGECT4']==2)]
sub2=sub1.copy()

#1.1.3 Coding out missing data (-1 = missing |
# No missing value for PPAGECT4 & PETHM)
sub2['W1_F1']=sub2['W1_F1'].replace(-1, numpy.nan) 

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
plt.xlabel('Ethnic Minority')
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
plt.xlabel('Ethnic Minority')
plt.ylabel('Percentage of Respondent Having Negative Outlook')

#######################################################################################################
#3. Post-hoc test for PPETHM v NEGATIVE
#3.1 3v2
recode2 = {'3_Hispanic': '3_Hispanic', '2_Black': '2_Black'}
sub2['COMP3v2']= sub2['PPETHM'].map(recode2)

# contingency table of observed counts
ct3v2=pandas.crosstab(sub2['NEGATIVE'], sub2['COMP3v2'])
print (ct3v2)
colsum=ct3v2.sum(axis=0)
colpct=ct3v2/colsum
print(colpct)

print ('chi-square value, p value, expected counts')
cs3v2= scipy.stats.chi2_contingency(ct3v2)
print (cs3v2)

############################################
#3.2 3v1
recode2 = {'3_Hispanic': '3_Hispanic', '1_White': '1_White'}
sub2['COMP3v1']= sub2['PPETHM'].map(recode2)

# contingency table of observed counts
ct3v1=pandas.crosstab(sub2['NEGATIVE'], sub2['COMP3v1'])
print (ct3v1)
colsum=ct3v1.sum(axis=0)
colpct=ct3v1/colsum
print(colpct)

print ('chi-square value, p value, expected counts')
cs3v1= scipy.stats.chi2_contingency(ct3v1)
print (cs3v1)

############################################
#3.3 3v4
recode2 = {'3_Hispanic': '3_Hispanic', '4_Other': '4_Other'}
sub2['COMP3v4']= sub2['PPETHM'].map(recode2)

# contingency table of observed counts
ct3v4=pandas.crosstab(sub2['NEGATIVE'], sub2['COMP3v4'])
print (ct3v4)
colsum=ct3v4.sum(axis=0)
colpct=ct3v4/colsum
print(colpct)

print ('chi-square value, p value, expected counts')
cs3v4= scipy.stats.chi2_contingency(ct3v4)
print (cs3v4)

############################################
#3.4 3v5
recode2 = {'3_Hispanic': '3_Hispanic', '5_2+Ethnic': '5_2+Ethnic'}
sub2['COMP3v5']= sub2['PPETHM'].map(recode2)

# contingency table of observed counts
ct3v5=pandas.crosstab(sub2['NEGATIVE'], sub2['COMP3v5'])
print (ct3v5)
colsum=ct3v5.sum(axis=0)
colpct=ct3v5/colsum
print(colpct)

print ('chi-square value, p value, expected counts')
cs3v5= scipy.stats.chi2_contingency(ct3v5)
print (cs3v5)

############################################
#3.5 2v1 - sig!!!
recode2 = {'2_Black': '2_Black', '1_White': '1_White'}
sub2['COMP2v1']= sub2['PPETHM'].map(recode2)

# contingency table of observed counts
ct2v1=pandas.crosstab(sub2['NEGATIVE'], sub2['COMP2v1'])
print (ct2v1)
colsum=ct2v1.sum(axis=0)
colpct=ct2v1/colsum
print(colpct)

print ('chi-square value, p value, expected counts')
cs2v1= scipy.stats.chi2_contingency(ct2v1)
print (cs2v1)

############################################
#3.6 2v4
recode2 = {'2_Black': '2_Black', '4_Other': '4_Other'}
sub2['COMP2v4']= sub2['PPETHM'].map(recode2)

# contingency table of observed counts
ct2v4=pandas.crosstab(sub2['NEGATIVE'], sub2['COMP2v4'])
print (ct2v4)
colsum=ct2v4.sum(axis=0)
colpct=ct2v4/colsum
print(colpct)

print ('chi-square value, p value, expected counts')
cs2v4= scipy.stats.chi2_contingency(ct2v4)
print (cs2v4)

############################################
#3.7 2v5
recode2 = {'2_Black': '2_Black', '5_2+Ethnic': '5_2+Ethnic'}
sub2['COMP2v5']= sub2['PPETHM'].map(recode2)

# contingency table of observed counts
ct2v5=pandas.crosstab(sub2['NEGATIVE'], sub2['COMP2v5'])
print (ct2v5)
colsum=ct2v5.sum(axis=0)
colpct=ct2v5/colsum
print(colpct)

print ('chi-square value, p value, expected counts')
cs2v5= scipy.stats.chi2_contingency(ct2v5)
print (cs2v5)

############################################
#3.8 1v4
recode2 = {'2_Black': '2_Black', '4_Other': '4_Other'}
sub2['COMP1v4']= sub2['PPETHM'].map(recode2)

# contingency table of observed counts
ct1v4=pandas.crosstab(sub2['NEGATIVE'], sub2['COMP1v4'])
print (ct1v4)
colsum=ct1v4.sum(axis=0)
colpct=ct1v4/colsum
print(colpct)

print ('chi-square value, p value, expected counts')
cs1v4= scipy.stats.chi2_contingency(ct1v4)
print (cs1v4)

############################################
#3.9 1v5
recode2 = {'2_Black': '2_Black', '5_2+Ethnic': '5_2+Ethnic'}
sub2['COMP1v5']= sub2['PPETHM'].map(recode2)

# contingency table of observed counts
ct1v5=pandas.crosstab(sub2['NEGATIVE'], sub2['COMP1v5'])
print (ct1v5)
colsum=ct1v5.sum(axis=0)
colpct=ct1v5/colsum
print(colpct)

print ('chi-square value, p value, expected counts')
cs1v5= scipy.stats.chi2_contingency(ct1v5)
print (cs1v5)

############################################
#3.10 4v5
recode2 = {'4_Other': '4_Other', '5_2+Ethnic': '5_2+Ethnic'}
sub2['COMP4v5']= sub2['PPETHM'].map(recode2)

# contingency table of observed counts
ct4v5=pandas.crosstab(sub2['NEGATIVE'], sub2['COMP4v5'])
print (ct4v5)
colsum=ct4v5.sum(axis=0)
colpct=ct4v5/colsum
print(colpct)

print ('chi-square value, p value, expected counts')
cs4v5= scipy.stats.chi2_contingency(ct4v5)
print (cs4v5)

#######################################################################################################
#Check for number in each ethnicity
seaborn.countplot(x="PPETHM", data=sub2)
plt.xlabel('Ethnic Minority')
plt.title('Ethnic Minority among person age 18-44 years in the OOL Study')