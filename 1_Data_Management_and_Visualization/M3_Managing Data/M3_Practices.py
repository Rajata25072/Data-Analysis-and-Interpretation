# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 00:06:44 2016

@author: DIMEX
"""

import pandas
import numpy

#import data
data = pandas.read_csv('ool_pds.csv', sep=',', low_memory=False)

# bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format', lambda x:'%f'%x)

#setting variables you will be working with to numeric and category
#Independent variable
data['PPAGECAT'] = data['PPAGECAT'].astype('category') #age categories
data['PPEDUCAT'] = data['PPEDUCAT'].astype('category') #factor variable - education level
data['PPETHM'] = data['PPETHM'].astype('category') #factor variable - ethnic
data['PPGENDER'] = data['PPGENDER'].astype('category') #factor variable
data['PPINCIMP'] = data['PPINCIMP'].astype('category') #factor variable - household income
data['PPMARIT'] = data['PPMARIT'].astype('category') #factor variable - marital status
data['W1_P2'] = data['W1_P2'].astype('category') #factor variable - social class belong to
data['W1_P3'] = data['W1_P3'].astype('category') #factor variable - social class family belong to
data['W1_F1'] = data['W1_F1'].astype('category') #factor variable 
#- When you think about your future, are you generally optimistic, pessimistic, or neither
#optimistic nor pessimistic? + A - extremely, moderately or slightly
data['W1_F2'] = data['W1_F2'].astype('category') #factor variable 
#And when you think about the future of the United States as a whole, are you
#generally optimistic, pessimistic, or neither optimistic nor pessimistic?


#practice subset data to non-white
sub1=data[(data['PPETHM']==2) | (data['PPETHM']==3) | (data['PPETHM']==4) | (data['PPETHM']==5)]

#make a copy of new subsetted data
sub2 = sub1.copy()

print 'counts for original W1_F1'
c1 = sub2['W1_F1'].value_counts(sort=False, dropna=False)
print(c1)

print 'counts for original W1_F2'
c2 = sub2['W1_F2'].value_counts(sort=False, dropna=False)
print(c2)

# recode missing values to python missing (NaN)
sub2['W1_F1']=sub2['W1_F1'].replace(-1, numpy.nan)
sub2['W1_F2']=sub2['W1_F2'].replace(-1, numpy.nan)

#if you want to include a count of missing add ,dropna=False after sort=False 
print 'counts for W1_F1 with -1 set to NAN and number of missing requested'
c2 = sub2['W1_F1'].value_counts(sort=False, dropna=False)
print(c2)

#coding in valid data
#recode missing values to numeric value, in this example replace NaN with 11
sub2['W1_F1'].fillna(11, inplace=True)
#recode 11 values as missing
sub2['W1_F1']=sub2['W1_F1'].replace(-1, numpy.nan)

print 'S2AQ8A with Blanks recoded as 11 and 99 set to NAN'
# check coding
chk2 = sub2['S2AQ8A'].value_counts(sort=False, dropna=False)
print(chk2)
ds2= sub2["S2AQ8A"].describe()
print(ds2)

#recoding values for W1_F1 into a new variable, W1_F1e (extremely - moderate - slightly)
data['W1_P3'] = data['W1_P3'].convert_objects(convert_numeric=True)
data['W1_F1'] = data['W1_F1'].convert_objects(convert_numeric=True)

#recode W1__F1 to repare for secondary data creation
recode1 = {1: 1, 2: 4, 3: 5}
sub2['W1_F1']= sub2['W1_F1'].map(recode1)

#recode W1__F2 to repare for secondary data creation
sub2['W1_F2']= sub2['W1_F2'] - 1


sub2['W1_F1e'] = sub2['W1_F1'] + sub2['W1_F2']

    

ds3= sub2["W1_F1e"].value_counts(sort=False, dropna=False)
print(ds3)