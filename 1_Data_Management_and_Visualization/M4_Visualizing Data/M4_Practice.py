# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 16:47:05 2016

@author: DIMEX
"""

#import library
import pandas
import numpy
import seaborn
import matplotlib.pyplot as plt

#import data
data = pandas.read_csv('ool_pds.csv', sep=',', low_memory=False)
data2 = pandas.read_csv('gapminder.csv', low_memory=False)

#setting variables to be working with to numeric and category
data['PPAGECT4'] = data['PPAGECT4'].astype('category') #factor variable - age category
data['PPETHM'] = data['PPETHM'].astype('category') #factor variable - ethnic
data['W1_D1'] = data['W1_D1'].convert_objects(convert_numeric=True) #numeric variable 
#- [Barack Obama] How would you rate
data['W1_D9'] = data['W1_D9'].convert_objects(convert_numeric=True) #numeric variable 
#- [Hillary Clinton] How would you rate
data['W1_P2'] = data['W1_P2'].convert_objects(convert_numeric=True) #factor variable - social class belong to
data['W1_P3'] = data['W1_P3'].convert_objects(convert_numeric=True) #factor variable - social class family belong to
data['W1_F1'] = data['W1_F1'].astype('category') #factor variable 
#Future outlook - generally optimistic, pessimistic, or neither
data['W1_F6'] = data['W1_F6'].convert_objects(convert_numeric=True) #factor variable - life outlook (quantitative)
#How far along the road to your American Dream do you think you will ultimately get
#on a 10-point scale where 1 is not far at all and 10 nearly there?

#setting variables you will be working with to numeric
data2['internetuserate'] = data2['internetuserate'].convert_objects(convert_numeric=True)
data2['urbanrate'] = data2['urbanrate'].convert_objects(convert_numeric=True)
data2['incomeperperson'] = data2['incomeperperson'].convert_objects(convert_numeric=True)
data2['hivrate'] = data2['hivrate'].convert_objects(convert_numeric=True)

#1.Make and implement data management decisions 
#No secondary variable
#1.1 Subset data
#1.1.1 Subset for selected variables
# subset variables in new data frame, sub1
sub1=data[['PPAGECT4','PPETHM', 'W1_D1', 'W1_D9','W1_P2', 'W1_P3', 'W1_F1', 'W1_F6']]

#1.1.2 Subset for young-middle-age (18-44) - PPAGECT4 of 1-2
sub1=sub1[(data['PPAGECT4']==1) | (data['PPAGECT4']==2)]
sub2=sub1.copy()

#1.1.3 Coding out missing data (-1 = missing |
# No missing value for PPAGECT4 & PETHM)
sub2['W1_P2']=sub2['W1_P2'].replace(-1, numpy.nan) 
sub2['W1_P3']=sub2['W1_P3'].replace(-1, numpy.nan) 
sub2['W1_F1']=sub2['W1_F1'].replace(-1, numpy.nan) 
sub2['W1_D1']=sub2['W1_D1'].replace(-1, numpy.nan) 
sub2['W1_D1']=sub2['W1_D1'].replace(998, numpy.nan) 
sub2['W1_D9']=sub2['W1_D9'].replace(-1, numpy.nan) 
sub2['W1_D9']=sub2['W1_D9'].replace(998, numpy.nan)
sub2['W1_F6']=sub2['W1_F6'].replace(-1, numpy.nan) 

data2['incomeperperson']=data2['incomeperperson'].replace(' ', numpy.nan)


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

#2. Data Exploration
print ("PPAGECT4")
print (sub2["PPAGECT4"].value_counts(sort=False))

print ("PPETHM")
print (sub2["PPETHM"].value_counts(sort=False))

print ("W1_P2")
print (sub2["W1_P2"].value_counts(sort=False))

print ("W1_P3")
print (sub2["W1_P3"].value_counts(sort=False))

print ("W1_F1n")
print (sub2["W1_F1n"].value_counts(sort=False))

print ("POSITIVE")
print (sub2["POSITIVE"].value_counts(sort=False))

print ("NEGATIVE")
print (sub2["NEGATIVE"].value_counts(sort=False))

#3. Practive Plotting Data
#3.1 Univariate bar graph for categorical variables
sub2["PPETHM"] = sub2["PPETHM"].astype('category')

seaborn.countplot(x="PPETHM", data=sub2)
plt.xlabel('Ethnic Minority')
plt.title('Ethnic Minority among person age 18-44 years in the OOL Study')

#3.2 Univariate histogram for quantitative variable:
seaborn.distplot(sub2["W1_D1"].dropna(), kde=False);
plt.xlabel('Rating Score for Barack Obama')
plt.title('Rating Score for Barack Obama among person age 18-44 years in the OOL Study')

# 3.3 Bivariate bar graph C->Q
seaborn.factorplot(x="PPETHM", y="W1_D1", data=sub2, kind="bar", ci=None)
plt.xlabel('Ethnic Minority')
plt.ylabel('Average Rating Score for Barck Obama')

# Check for correctness of W1_D1 proportion
sub3 = sub2[['W1_D1', 'PPETHM']].dropna()
print ('means for Rating Score for Barck Obama by ethnicity')
m1= sub3.groupby('PPETHM').mean()
print (m1) #correct

# 3.4 Bivariate bar graph C->C
seaborn.factorplot(x='PPETHM', y='NEGATIVE', data=sub2, kind="bar", ci=None)
plt.xlabel('Ethnic Minority')
plt.ylabel('Proportion of Negative Outlook')

# Check for correctness of negative proportion
sub4 = sub2[['NEGATIVE', 'PPETHM']].dropna()
print ('means for NEGATIVE by ethnicity')
m2 = sub4.groupby('PPETHM').mean()
print (m2) #correct

#3.5 basic scatterplot:  Q->Q

#3.5.1 Obama vs Hilary score
scat1 = seaborn.regplot(x="W1_D1", y="W1_D9", data=sub2)
plt.xlabel('Rating Score for Barack Obama')
plt.ylabel('Rating Score for Hillary Clinton')
plt.title('Scatterplot for the Association Between Obama and Hillary Rating Score') #somewhat positive relationship

#3.5.2 Obama score vs Life Outlook
scat2 = seaborn.regplot(x="W1_D1", y="W1_F6", data=sub2)
plt.xlabel('Rating Score for Barack Obama')
plt.ylabel('Rating Score for Life Outllok')
plt.title('Scatterplot for the Association Between President Likeness and Life Outlook') #no clear relationship

#3.5.3 Urban Rate vs Internet User Rate
scat3 = seaborn.regplot(x="urbanrate", y="internetuserate", fit_reg=False, data=data2)
plt.xlabel('Urban Rate')
plt.ylabel('Internet Use Rate')
plt.title('Scatterplot for the Association Between Urban Rate and Internet Use Rate') #positive relationship