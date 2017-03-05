# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 03:28:18 2016

@author: DIMEX
"""

#import library
import pandas
import numpy

#import data
data = pandas.read_csv('ool_pds.csv', sep=',', low_memory=False)

#setting variables to be working with to numeric and category
data['PPAGECT4'] = data['PPAGECT4'].astype('category') #factor variable - age category
data['PPETHM'] = data['PPETHM'].astype('category') #factor variable - ethnic
data['W1_P2'] = data['W1_P2'].convert_objects(convert_numeric=True) #factor variable - social class belong to
data['W1_P3'] = data['W1_P3'].convert_objects(convert_numeric=True) #factor variable - social class family belong to
data['W1_F1'] = data['W1_F1'].astype('category') #factor variable 
#Future outlook - generally optimistic, pessimistic, or neither

#1.Make and implement data management decisions 
#No secondary variable
#1.1 Subset data
#1.1.1 Subset for selected variables
# subset variables in new data frame, sub1
sub1=data[['PPAGECT4','PPETHM', 'W1_P2', 'W1_P3', 'W1_F1']]

#1.1.2 Subset for young-middle-age (18-44) - PPAGECT4 of 1-2
sub1=sub1[(data['PPAGECT4']==1) | (data['PPAGECT4']==2)]
sub2=sub1.copy()

#1.1.3 Coding out missing data (-1 = missing |
# No missing value for PPAGECT4 & PETHM)
sub2['W1_P2']=sub2['W1_P2'].replace(-1, numpy.nan) 
sub2['W1_P3']=sub2['W1_P3'].replace(-1, numpy.nan) 
sub2['W1_F1']=sub2['W1_F1'].replace(-1, numpy.nan) 

#1.1.4 Recoding variable
#recoding values for W1_F1 into a new variable, W1_F1n
#to be more intuitive (-1=negative,0=neither,1=positive)
recode1 = {1: 1, 2: 0, 3: -1}
sub2['W1_F1n']= sub2['W1_F1'].map(recode1)
type (sub2['W1_F1n'])

#1.1.5 Create secondary variable
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

#3) CLASSDIF the see the change in social class (-1 = decrease | 0 = same | 1 =increase)
def CLASSDIF (row):
   if row['W1_P2'] < row['W1_P3'] : 
      return -1 #personal class < family class (doing worse than family)
   elif row['W1_P2'] == row['W1_P3'] :
      return 0 #personal class = family class (stay the same)
   elif row['W1_P2'] > row['W1_P3'] :
      return 1 #personal class > family class (doing better than family)
sub2['CLASSDIF'] = sub2.apply (lambda row: CLASSDIF (row),axis=1)


#2. Data Exploration
print  ('=====================Data Exploration=======================')
#2.1 Before vs after coding missing data (-1 -> NaN)
print  ('------Before vs after coding missing data (-1 -> NaN)------')
#2.1.1 W1_P2 - before vs after
print ('counts for W1_P2 before - social class belong to')
b1 = sub1['W1_P2'].value_counts(sort=False)
print (b1)

print ('counts for W1_P2 after - social class belong to')
a1 = sub2['W1_P2'].value_counts(sort=False, dropna=False)
print (a1) #-1 become NaN

#2.1.2 W1_P3 - before vs after
print ('counts for W1_P3 before - social class family belong to')
b2 = sub1['W1_P3'].value_counts(sort=False)
print (b2)

print ('counts for W1_P3 after - social class family belong to')
a2 = sub2['W1_P3'].value_counts(sort=False, dropna=False)
print (a2) #-1 become NaN

#2.2 Before vs after coding out missing data+recoding
#-1 -> NaN | 1 -> 1 | 2 -> 0 | 3 -> -1
print ('------Before vs after coding out missing data+recoding------')
print ('----------(-1 -> NaN | 1 -> 1 | 2 -> 0 | 3 -> -1)----------')
print ('counts for W1_F1 before - Future outlook')
b2 = sub1['W1_F1'].value_counts(sort=False)
print (b2)

print ('counts for W1_F1n after - Future outlook')
a2 = sub2['W1_F1n'].value_counts(sort=True, dropna=False)
print (a2) #-1 become NaN

print ('percentage for W1_F1n after - Future outlook)')
a2n = sub2['W1_F1n'].value_counts(sort=True, dropna=False, normalize=True)
print (a2n) #-1 become NaN

#2.3 Before vs after grouping variable (Check if POSITIVE & NEGATIVE is right)
print ('------Explore secondary variable (POSITIVE & NEGATIVE)------')
print ('----------(Check if POSITIVE & NEGATIVE is right)----------')
print ('counts for W1_F1n - Future outlook (Positive=1 | Negative=-1)')
b3 = sub2['W1_F1n'].value_counts(sort=True, dropna=False)
print (b3)

print ('counts for POSITIVE (Positive=1)')
a3 = sub2['POSITIVE'].value_counts(sort=False, dropna=False)
print (a3)

print ('percentage for POSITIVE (Positive=1)')
a3n = sub2['POSITIVE'].value_counts(sort=False, dropna=False, normalize=True)
print (a3n)

print ('counts for NEGATIVE (Negative=1)')
a4 = sub2['NEGATIVE'].value_counts(sort=False, dropna=False)
print (a4)

print ('percentage for NEGATIVE (Negative=1)')
a4n = sub2['NEGATIVE'].value_counts(sort=False, dropna=False, normalize=True)
print (a4n)

#2.4 Explore secondary variable (CLASSDIF)
print ('----------Explore secondary variable (CLASSDIF)----------')
print ('counts for CLASSDIF (Self<family= -1, Self>family=1)')
a5 = sub2['CLASSDIF'].value_counts(sort=True, dropna=False)
print (a5)

print ('percentage for CLASSDIF (Self<family= -1, Self>family=1)')
a5n = sub2['CLASSDIF'].value_counts(sort=True, dropna=False, normalize=True)
print (a5n)