# -*- coding: utf-8 -*-

"""
Created on Sun Aug 30 10:50:43 2015

@author: smallpie
"""

#import library
import pandas
import numpy

#import data
data = pandas.read_csv('ool_pds.csv', sep=',', low_memory=False)

#check data
print (len(data)) #number of observations (rows)
print (len(data.columns)) # number of variables (columns)

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

#Dependent variable
# 1. When you think about your future, are you generally optimistic, pessimistic, or neither
#    optimistic nor pessimistic? + A - extremely, moderately or slightly
data['W1_F1'] = data['W1_F1'].astype('category') #factor variable - 
data['W1_F1A'] = data['W1_F1A'].astype('category') #factor variable - 
# 2. When you think about your future of USA, are you generally optimistic, pessimistic, or neither
#    optimistic nor pessimistic? + A - extremely, moderately or slightly
data['W1_F2'] = data['W1_F1'].astype('category') #factor variable - 
data['W1_F2A'] = data['W1_F1A'].astype('category') #factor variable - 
# 3. A basic American belief has been that if you work hard you can get ahead and reach
# the goals you set and more. Is this true or false today?
data['W1_F3'] = data['W1_F3'].astype('category') #factor variable - 
# 4. Now thinking about the country's economy, would you say that compared to one year
# ago, the nation's economy is now better, about the same, or worse?
data['W1_G2'] = data['W1_G2'].astype('category') #factor variable - 
# 5. Do you think the economy, in the country as a whole, will be better, about the same,
# or worse in 12 months?
data['W1_G3A'] = data['W1_G3A'].astype('category') #factor variable - 

#counts and percentages (i.e. frequency distributions) for each variable
c1 = data['PPETHM'].value_counts(sort=False) #count
print (c1)
p1 = data['PPETHM'].value_counts(sort=False, normalize=True) #percentage count
print (p1)

c2 = data['W1_P2'].value_counts(sort=False) 
print(c2)
p2 = data['W1_P2'].value_counts(sort=False, normalize=True) 
print (p2)

c3 = data['W1_F1'].value_counts(sort=False)
print(c3)
p3 = data['W1_F1'].value_counts(sort=False, normalize=True)
print (p3)

c4 = data['W1_G3A'].value_counts(sort=False)
print(c4)
p4 = data['W1_G3A'].value_counts(sort=False, normalize=True)
print (p4)


#ADDING TITLES
print ('counts for PPETHM')
c1 = data['PPETHM'].value_counts(sort=False)
print (c1)

print ('percentages for PPETHM')
p1 = data['PPETHM'].value_counts(sort=False, normalize=True)
print (p1)

print ('counts for W1_P2')
c2 = data['W1_P2'].value_counts(sort=False)
print(c2)

print ('percentages for W1_P2')
p2 = data['W1_P2'].value_counts(sort=False, normalize=True)
print (p2)

print ('counts for W1_F1')
c3 = data['W1_F1'].value_counts(sort=False, dropna=False)
print(c3)

print ('percentages for W1_F1')
p3 = data['W1_F1'].value_counts(sort=False, normalize=True)
print (p3)

print ('counts for W1_G3A')
c4 = data['W1_G3A'].value_counts(sort=False, dropna=False)
print(c4)

print ('percentages for W1_G3A')
p4 = data['W1_G3A'].value_counts(sort=False, dropna=False, normalize=True)
print (p4)

#ADDING MORE DESCRIPTIVE TITLES
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

print ('counts for W1_G3A - country economy outlook (Optimistic=1|Neither=2|Pessimistic=3)')
c4 = data['W1_G3A'].value_counts(sort=False, dropna=False)
print(c4)

print ('percentages for W1_G3A - country economy outlook (Optimistic=1|Neither=2|Pessimistic=3)')
p4 = data['W1_G3A'].value_counts(sort=False, dropna=False, normalize=True)
print (p4)


# freqeuncy disributions using the 'bygroup' function
ct1= data.groupby('PPETHM').size()
print (ct1)

pt1 = data.groupby('PPETHM').size() * 100 / len(data)
print (pt1)

ct2= data.groupby('W1_G3A').size()
print (ct2)

pt2 = data.groupby('W1_G3A').size() * 100 / len(data)
print (pt2)

ct3= data.groupby('W1_F1').size()
print (ct3)

pt3 = data.groupby('W1_F1').size() * 100 / len(data)
print (pt3)

ct4= data.groupby('W1_G3A').size()
print (ct4)

pt4 = data.groupby('W1_G3A').size() * 100 / len(data)
print (pt4)

#subset data to young adults + wokring age with age 18 to 54 who have smoked in the past 12 months
sub1=data[(data['PPAGECAT'] == 1) | (data['PPAGECAT'] == 2) | (data['PPAGECAT'] == 3) | (data['PPAGECAT'] == 4)] 

#make a copy of my new subsetted data
sub2 = sub1.copy()


# frequency distritions on new sub2 data frame
print ('counts for PPETHM')
c5 = sub2['PPETHM'].value_counts(sort=False)
print(c5)

print ('percentages for PPETHM')
p5 = sub2['PPETHM'].value_counts(sort=False, normalize=True)
print (p5)

print ('counts for W1_G3A')
c6 = sub2['W1_G3A'].value_counts(sort=False)
print(c6)

print ('percentages for W1_G3A')
p6 = sub2['W1_G3A'].value_counts(sort=False, normalize=True)
print (p6)

#upper-case all DataFrame column names - place afer code for loading data above
data.columns = map(str.upper, data.columns)

# bug fix for display formats to avoid run time errors - put after code for loading data above
pandas.set_option('display.float_format', lambda x:'%f'%x)


sub1[1,:]