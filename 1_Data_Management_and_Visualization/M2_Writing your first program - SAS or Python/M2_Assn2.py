# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 21:25:36 2016

@author: DIMEX
"""

#import library
import pandas
import numpy

#import data
data = pandas.read_csv('ool_pds.csv', sep=',', low_memory=False)

#setting variables you will be working with to numeric and category
data['PPETHM'] = data['PPETHM'].astype('category') #factor variable - ethnic
data['W1_P2'] = data['W1_P2'].astype('category') #factor variable - social class belong to
data['W1_F1'] = data['W1_F1'].astype('category') #factor variable 
#- When you think about your future, are you generally optimistic, pessimistic, or neither
#optimistic nor pessimistic? + A - extremely, moderately or slightly

print ('counts for PPETHM - ethinicity (White=1/Black=2/Other=3/Hispanic=4/2+=5)')
c1 = data['PPETHM'].value_counts(sort=False)
print (c1)

print ('percentages for PPETHM - ethinicity (White=1/Black=2/Other=3/Hispanic=4/2+=5)')
p1 = data['PPETHM'].value_counts(sort=False, normalize=True)
print (p1)

print ('counts for W1_P2 - social class belong to (Poor=1/Working=2/Middle=3/Upper-Mid=4/Upper=5)')
c2 = data['W1_P2'].value_counts(sort=False)
print(c2)

print ('percentages for W1_P2 - social class belong to (Poor=1/Working=2/Middle=3/Upper-Mid=4/Upper=5)')
p2 = data['W1_P2'].value_counts(sort=False, normalize=True)
print (p2)

print ('counts for W1_F1 - their onw future outlook (Optimistic=1|Neither=2|Pessimistic=3)')
c3 = data['W1_F1'].value_counts(sort=False, dropna=False)
print(c3)

print ('percentages for W1_F1 - their onw future outlook (Optimistic=1|Neither=2|Pessimistic=3)')
p3 = data['W1_F1'].value_counts(sort=False, normalize=True)
print (p3)
