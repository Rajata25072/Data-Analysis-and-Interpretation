# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 16:01:49 2016

@author: DIMEX
"""
#######################################################
# 1. import library & data
#######################################################
# 1.1 import library 
import numpy
import pandas
import scipy.stats
import statsmodels.formula.api as smf 
import seaborn
import matplotlib.pyplot as plt

# 1.2 import data
data = pandas.read_csv('ool_pds.csv', sep=',', low_memory=False)

# 1.3 setting variables to be working with to numeric and category
# 1.3.1 R-square var. - Obama Score v Hillary Score | Obama Score v Social Class??
data['W1_D1'] = data['W1_D1'].convert_objects(convert_numeric=True) 
#numeric variable - [Barack Obama] How would you rate
data['W1_P2'] = data['W1_P2'].convert_objects(convert_numeric=True) 
#factor variable - social class belong to

# 1.3.2 Moderator - Age Group
data['PPAGECT4'] = data['PPAGECT4'].astype('category') #factor variable - age category

#######################################################
# 2. Make and implement data management decisions 
#######################################################
# 2.1 Subset data for selected variables in new data frame, sub1
sub1=data[['W1_D1','W1_P2','PPAGECT4']]

# 2.2 Coding out missing data (-1 = missing |
# No missing value for PPAGECT4
sub1['W1_D1']=sub1['W1_D1'].replace(-1, numpy.nan) 
sub1['W1_D1']=sub1['W1_D1'].replace(998, numpy.nan) 
sub1['W1_P2']=sub1['W1_P2'].replace(-1, numpy.nan) 

# 2.3 Creating secondary variable: Age to 2 groups
recode3 = {1: '1_Young+Middle', 2: '1_Young+Middle', 3: '2_Elder', 4: '2_Elder'}
sub1['AGROUP']= sub1['PPAGECT4'].map(recode3)
sub1['AGROUP']=sub1['AGROUP'].astype('category')

# 2.4 Seperate data by moderator: AGROUP
sub2=sub1[(sub1['AGROUP']=='1_Young+Middle')]
sub3=sub1[(sub1['AGROUP']=='2_Elder')]

#######################################################
# 3. Analysis   - Correlation - Obama Score v Social Class w/ moderator = AGE
#######################################################
# 3.1 Get rid of NaN
sub21 = sub2[['W1_D1', 'W1_P2']].dropna()
sub31 = sub3[['W1_D1', 'W1_P2']].dropna()

# 3.2 Correlation Analyses
print ('Association between social class and Obama score for YOUNG&MIDDLE-AGE person')
print (scipy.stats.pearsonr(sub21['W1_D1'], sub21['W1_P2'])) #signicant negative relationship
#(-0.10298238865729449, 0.0035043868801347246)

print ('Association between social class and Obama score for ELDER person')
print (scipy.stats.pearsonr(sub31['W1_D1'], sub31['W1_P2'])) #insignificant relationship
#(-0.040199742829663196, 0.14269823886145297)

# 3.3 Scatterplot
scat1 = seaborn.regplot(x="W1_P2", y="W1_D1", data=sub21)
plt.xlabel('Social class')
plt.ylabel('Obama score')
plt.title('Scatterplot for the association between social class and Obama score for YOUNG&MIDDLE-AGE person')
print (scat1)

scat2 = seaborn.regplot(x="W1_P2", y="W1_D1", data=sub31)
plt.xlabel('Social class')
plt.ylabel('Obama score')
plt.title('Scatterplot for the association between social class and Obama score for ELDER person')
print (scat2)


#######################################################

# lEFTOVER

#######################################################
# 3.3 Correlation - Obama Score v Hillary Score w/ moderator  = PPETHM
# 3.3.1 Get rid of NaN
sub23 = sub2[['W1_D1', 'W1_D9']].dropna()
sub33 = sub3[['W1_D1', 'W1_D9']].dropna()
sub43 = sub4[['W1_D1', 'W1_D9']].dropna()

# 3.3.2 Correlation Analyses
print ('Association between Obama and Hillary rating score for LOWER social class')
print (scipy.stats.pearsonr(sub23['W1_D1'], sub23['W1_D9']))

print ('Association between Obama and Hillary rating score for MIDDLE social class')
print (scipy.stats.pearsonr(sub33['W1_D1'], sub33['W1_D9']))

print ('Association between Obama and Hillary rating score for UPPER social class')
print (scipy.stats.pearsonr(sub43['W1_D1'], sub43['W1_D9']))

# 3.3.3 Scatterplot
scat1 = seaborn.regplot(x="W1_D9", y="W1_D1", data=sub23)
plt.xlabel('Hillary score')
plt.ylabel('Obama score')
plt.title('Scatterplot for the association between Obama and Hillary rating score for LOWER social class')
print (scat1)

scat2 = seaborn.regplot(x="W1_D9", y="W1_D1", data=sub33)
plt.xlabel('Hillary score')
plt.ylabel('Obama score')
plt.title('Scatterplot for the association between Obama and Hillary rating score for MIDDLE social class')
print (scat2)

scat3 = seaborn.regplot(x="W1_D9", y="W1_D1", data=sub43)
plt.xlabel('Hillary score')
plt.ylabel('Obama score')
plt.title('Scatterplot for the association between Obama and Hillary rating score for UPPER social class')
print (scat3)

#######################################################
# 3.3 Correlation - Obama Score v Life Outlook w/ moderator  = PPETHM
# 3.3.1 Get rid of NaN
sub23 = sub2[['W1_D1', 'W1_F6']].dropna()
sub33 = sub3[['W1_D1', 'W1_F6']].dropna()
sub43 = sub4[['W1_D1', 'W1_F6']].dropna()

# 3.3.2 Correlation Analyses
print ('Association between Obama and Hillary rating score for LOWER social class')
print (scipy.stats.pearsonr(sub23['W1_D1'], sub23['W1_D9']))

print ('Association between Obama and Hillary rating score for MIDDLE social class')
print (scipy.stats.pearsonr(sub33['W1_D1'], sub33['W1_D9']))

print ('Association between Obama and Hillary rating score for UPPER social class')
print (scipy.stats.pearsonr(sub43['W1_D1'], sub43['W1_D9']))

# 3.3.3 Scatterplot
scat1 = seaborn.regplot(x="W1_F6", y="W1_D1", data=sub23)
plt.xlabel('Hillary score')
plt.ylabel('Life Outlook')
plt.title('Scatterplot for the association between Obama score and life outlook for LOWER social class')
print (scat1)

scat2 = seaborn.regplot(x="W1_F6", y="W1_D1", data=sub33)
plt.xlabel('Hillary score')
plt.ylabel('Life Outlook')
plt.title('Scatterplot for the association between Obama score and life outlook for MIDDLE social class')
print (scat2)

scat3 = seaborn.regplot(x="W1_F6", y="W1_D1", data=sub43)
plt.xlabel('Hillary score')
plt.ylabel('Life Outlook')
plt.title('Scatterplot for the association between Obama score and life outlook for UPPER social class')
print (scat3)

# 1.3.1 recoding values for W1_F1 into a new variable, W1_F1n
#to be more intuitive (-1=negative,0=neither,1=positive)
recode1 = {1: 1, 2: 0, 3: -1}
sub1['W1_F1n']= sub1['W1_F1'].map(recode1)
type (sub1['W1_F1n'])

# 1.3.2 recoding ethnicity to be more intuituve
#White=1/Black=2/Other=3/Hispanic=4/2+=5
recode2 = {1: '1_White', 2: '2_Black', 3: '4_Other', 4: '3_Hispanic', 5: '5_2+Ethnic'}
sub1['PPETHM']= sub1['PPETHM'].map(recode2)

# 1.3.3 recoding social class to 3 groups
recode3 = {1: '1_Lower', 2: '1_Lower', 3: '2_Middle', 4: '3_Upper', 5: '3_Upper'}
sub1['SOCIAL']= sub1['W1_P2'].map(recode3)
sub1['SOCIAL']=sub1['SOCIAL'].astype('category')

# 0.3.1 ANOVA var. - PPETHM v Obama Score
data['PPETHM'] = data['PPETHM'].astype('category') #factor variable - ethnic

# 0.3.2 Chi-square var. - PPETHM v NEGATIVE
data['W1_F1'] = data['W1_F1'].astype('category') #factor variable 
#Future outlook - generally optimistic, pessimistic, or neither

data['W1_D9'] = data['W1_D9'].convert_objects(convert_numeric=True) #numeric variable 
#- [Hillary Clinton] How would you rate
data['W1_F6'] = data['W1_F6'].convert_objects(convert_numeric=True) #factor variable - life outlook (quantitative)
#How far along the road to your American Dream do you think you will ultimately get
#on a 10-point scale where 1 is not far at all and 10 nearly there?

# 1.4 Create secondary variable: NEGATIVE outlook
def NEGATIVE (row):
   if row['W1_F1n'] == -1 : 
      return 1 
   elif (row['W1_F1n'] == 0) | (row['W1_F1n'] == 1) :
      return 0    
   else :
       return numpy.nan     
sub1['NEGATIVE'] = sub1.apply (lambda row: NEGATIVE (row),axis=1)

sub2=sub1[(sub1['SOCIAL']=='1_Lower')]
sub3=sub1[(sub1['SOCIAL']=='2_Middle')]
sub4=sub1[(sub1['SOCIAL']=='3_Upper')]


# 3.1 ANOVA
# 3.1.1 Get rid on NaN
sub21 = sub2[['W1_D1', 'PPETHM']].dropna()
sub31 = sub3[['W1_D1', 'PPETHM']].dropna()
sub41 = sub4[['W1_D1', 'PPETHM']].dropna()

# 3.1.2 Mean comparison
print ("Means for Obama rating score by Ethnicity for lower social class")
m1= sub21.groupby('PPETHM').mean()
print (m1)

print ("Means for Obama rating score by Ethnicity for middle social class")
m2= sub31.groupby('PPETHM').mean()
print (m2)

print ("Means for Obama rating score by Ethnicity for upper social class")
m3= sub41.groupby('PPETHM').mean()
print (m3)

# 3.1.3 ANOVA Analysis
print ('Association between ethnicity and Obama rating score for those in LOWER social class')
model1 = smf.ols(formula='W1_D1 ~ C(PPETHM)', data=sub21).fit()
print (model1.summary())

print ('Association between ethnicity and Obama rating score for those in MIDDLE social class')
model2 = smf.ols(formula='W1_D1 ~ C(PPETHM)', data=sub31).fit()
print (model2.summary())

print ('Association between ethnicity and Obama rating score for those in UPPER social class')
model3 = smf.ols(formula='W1_D1 ~ C(PPETHM)', data=sub41).fit()
print (model3.summary())

# 3.1.4 bivariate bar graph
seaborn.factorplot(x="PPETHM", y="W1_D1", data=sub21, kind="bar", ci=None,
                   order=["2_Black", "3_Hispanic", "4_Other", "1_White", "5_2+Ethnic"])
plt.xlabel('Ethnicity')
plt.ylabel('Obama rating score')

seaborn.factorplot(x="PPETHM", y="W1_D1", data=sub31, kind="bar", ci=None,
                   order=["2_Black", "4_Other", "5_2+Ethnic", "3_Hispanic", "1_White"])
plt.xlabel('Ethnicity')
plt.ylabel('Obama rating score')

seaborn.factorplot(x="PPETHM", y="W1_D1", data=sub41, kind="bar", ci=None,
                   order=["2_Black", "3_Hispanic", "4_Other", "1_White", "5_2+Ethnic"])
plt.xlabel('Ethnicity')
plt.ylabel('Obama rating score')

# Summarize - order change
# 1. Same ranking in Lower and Upper with difference in hispanic and other ethnic score
# 2. Middle class has different ranking

#######################################################