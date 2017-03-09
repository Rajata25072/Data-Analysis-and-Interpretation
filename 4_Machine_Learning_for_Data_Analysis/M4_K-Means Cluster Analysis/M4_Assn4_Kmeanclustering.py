# -*- coding: utf-8 -*-
"""
The K-means clustering Assignment which aims to group, or cluster, observations 
into subsets based on their similarity of responses on multiple variables. 

@author: smallpi
"""

#######################################################
# 1. import library & data
#######################################################
# 1.1 import library 
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
import os

# 1.2 set directory
os.chdir("D:\!MOOC\Python_Directory\Data Analysis and Interpretation")

# 1.3 import data
data = pd.read_csv('OOL_pds.csv', sep=',', low_memory=False)
data.columns = map(str.upper, data.columns) #upper-case all DataFrame column names
data.dtypes
data.describe()

# 1.4 Setting Clustering variables
data['OOL'] = data['W1_F6'].convert_objects(convert_numeric=True) 
data['SEX'] = data['PPGENDER'].astype('category') # ft var. (binary) - gender
data['AGE'] = pd.to_numeric(data['PPAGE'], errors='coerce') # qt var. - age
data['EDU'] = pd.to_numeric(data['PPEDUCAT'], errors='coerce') # ft var. (can order) - education
data['INC'] = pd.to_numeric(data['PPINCIMP'], errors='coerce') # qt var. - household income
data['MARITAL'] = data['PPMARIT'].astype('category') #factor variable - marital status
data['ETHM'] = data['PPETHM'].astype('category')  # ft var. (can't order) - ethnic
data['SOC'] = pd.to_numeric(data['W1_P2'], errors='coerce') # ft var. (can order) - social class 
data['SOC_F'] = data['W1_P3'].convert_objects(convert_numeric=True) #factor variable - social class family belong to
data['UNEMP'] = data['W1_P11'].astype('category') # ft var. (binary) - unemploy
data['OBAMA'] = data['W1_D1'].convert_objects(convert_numeric=True) #rating of [Barack Obama] 
data['HILLARY'] = data['W1_D9'].convert_objects(convert_numeric=True) #rating of [Hillary Clinton] 


#######################################################
# 2. Make and implement data management decisions 
#######################################################
# 2.1 Subset for selected variables
s_data=data[['OOL','SEX','AGE','EDU','INC','MARITAL','ETHM','SOC','SOC_F','UNEMP','OBAMA','HILLARY']]
s_data.dtypes
s_data.describe() #describe quantitative var.

# 2.2 Coding out missing data (-1 = missing)
# No missing value for SEX & AGE & EDU & INC & MARITAL & ETHM 
s_data['OOL'] = s_data['OOL'].replace(-1, np.nan) 
s_data['SOC'] = s_data['SOC'].replace(-1, np.nan) 
s_data['SOC_F'] = s_data['SOC_F'].replace(-1, np.nan) 
s_data['UNEMP'] = s_data['UNEMP'].replace(-1, np.nan) 
s_data['OBAMA'] = s_data['OBAMA'].replace(-1, np.nan) 
s_data['OBAMA'] = s_data['OBAMA'].replace(998, np.nan) 
s_data['HILLARY'] = s_data['HILLARY'].replace(-1, np.nan) 
s_data['HILLARY'] = s_data['HILLARY'].replace(998, np.nan) 

data_clean = s_data.dropna()
data_clean.dtypes
data_clean.describe()

# 2.3 Recode variables
# 2.3.1 Recode inc to be quantitative
recode1 = {1:2500, 2:6250, 3:8750, 4:11250, 5:13750, 6: 17500, 7:22500, 8: 27500, 9: 32500, 
           10:37500, 11: 45000, 12:55000, 13:67500, 14: 80000, 15: 92500, 16:112500, 
           17: 137500, 18: 162500, 19: 200000}
print (data_clean['INC'].value_counts(sort=False)) #before recoding
data_clean['INC']= data_clean['INC'].map(recode1)
print (data_clean['INC'].value_counts(sort=False)) #after recoding

# 2.3.3 Recode soc to start with 0
print (data_clean['SOC'].value_counts(sort=False)) #before recoding
data_clean['SOC'] = data_clean['SOC'] -1
print (data_clean['SOC'].value_counts(sort=False)) #after recoding

# 2.3.4 Recode sex to have 0 = male | 1 = female
recode1 = {1: 0, 2: 1}
data_clean['SEX']= data_clean['SEX'].map(recode1)
data_clean['SEX'] = data_clean['SEX'].astype('category') 
print (data_clean['SEX'].value_counts(sort=False))

# 2.3.5 Recode unemp to have 0 = employ | 1 = unemploy
recode1 = {1: 1, 2: 0}
data_clean['UNEMP']= data_clean['UNEMP'].map(recode1)
data_clean['UNEMP'] = data_clean['UNEMP'].astype('category')
print (data_clean['UNEMP'].value_counts(sort=False))

# 2.4 Check the data
data_clean.dtypes
data_clean.describe()

# 2.5 Set prediction and target variable
cluster = data_clean[['SEX','AGE','EDU','INC','MARITAL','ETHM','SOC','SOC_F','UNEMP','OBAMA','HILLARY']]

# 2.6 standardize predictors to have mean=0 and sd=1
clustervar=cluster.copy()
clustervar['SEX']=preprocessing.scale(clustervar['SEX'].astype('float64'))
clustervar['AGE']=preprocessing.scale(clustervar['AGE'].astype('float64'))
clustervar['EDU']=preprocessing.scale(clustervar['EDU'].astype('float64'))
clustervar['INC']=preprocessing.scale(clustervar['INC'].astype('float64'))
clustervar['MARITAL']=preprocessing.scale(clustervar['MARITAL'].astype('float64'))
clustervar['ETHM']=preprocessing.scale(clustervar['ETHM'].astype('float64'))
clustervar['SOC']=preprocessing.scale(clustervar['SOC'].astype('float64'))
clustervar['SOC_F']=preprocessing.scale(clustervar['SOC_F'].astype('float64'))
clustervar['UNEMP']=preprocessing.scale(clustervar['UNEMP'].astype('float64'))
clustervar['OBAMA']=preprocessing.scale(clustervar['OBAMA'].astype('float64'))
clustervar['HILLARY']=preprocessing.scale(clustervar['HILLARY'].astype('float64'))

# 2.7 Check the data
clustervar.dtypes
clustervar.describe()

# 2.8 Split into training and testing sets
clus_train, clus_test =   train_test_split(clustervar, test_size=.3, random_state=123)

print (clus_train.shape)
print (clus_test.shape)


##############################################################################
# 3. Perform Analysis
##############################################################################
# 3.1 Calculate distance for k-means cluster analysis for 1-9 clusters
from scipy.spatial.distance import cdist
clusters=range(1,10)
meandist=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(clus_train)
    clusassign=model.predict(clus_train)
    meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1)) 
    / clus_train.shape[0]) #use minimum cluster distance from point to cluster centroid as distance

# 3.2 Plot average distance from observations from the cluster centroid
# To use the Elbow Method to identify number of clusters to choose
plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method') #k should be 2,3,4

# 3.3 Compare different k cluster (assign cluster & plot PCA of 2 dimensions)
# 3.3.1 Prepare PCA of 2 dimension
from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(clus_train)

# 3.3.2 k = 2
model2=KMeans(n_clusters=2)
model2.fit(clus_train)
clusassign=model2.predict(clus_train)

plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model2.labels_,)
plt.xlabel('Canonical variable 1'); plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 2 Clusters')
plt.show()

# 3.3.3 k = 3 -> chosen
model3=KMeans(n_clusters=3)
model3.fit(clus_train)
clusassign=model3.predict(clus_train)

plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model3.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 3 Clusters')
plt.show()

# 3.3.4 k = 4 -> worst!
model4=KMeans(n_clusters=4)
model4.fit(clus_train)
clusassign=model4.predict(clus_train)

plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model4.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 4 Clusters')
plt.show()

# 3.4 Examine cluster variable means by cluster
# 3.4.1 Create a unique id variable from the index for the 
# cluster training data to merge with the cluster assignment variable
clus_train.reset_index(level=0, inplace=True) #reset index

# 3.4.2 Create a list that has the new index variable
cluslist=list(clus_train['index'])

# 3.4.3 Create a list of cluster assignments (Choose k = 3)
labels=list(model3.labels_)

# 3.4.4 Combine index variable list with cluster assignment list into a dictionary
newlist=dict(zip(cluslist, labels))

# 3.4.5 Convert newlist dictionary to a dataframe
newclus=DataFrame.from_dict(newlist, orient='index')
newclus.columns = ['cluster'] # rename the cluster assignment column

# 3.4.6 Reset newclus id to merge back with actual id in clus_train
newclus.reset_index(level=0, inplace=True)

# 3.4.7 Merge the cluster assignment dataframe with the cluster training set
# by the index variable
merged_train=pd.merge(clus_train, newclus, on='index')
merged_train.head(n=100)

# 3.4.8 Check cluster size
merged_train.cluster.value_counts()

# 3.4.9 Check variable means by cluster
clustergrp = merged_train.groupby('cluster').mean()
print ("Clustering variable means by cluster")
print(clustergrp)
# Cluster 0 -> Republican? -> mid-income | WHITE | aganst OBAMA & HILLARY
# Cluster 1 -> Low income -> | low EDU | low INC | MARITAL?? | low SOC & SOC_F | UNEMP
# Cluster 2 -> Rich Democrat -> Oldest | High EDU | high INC | high SOC 7 SOC_F | EMP | Favor OBAMA & HILLARY


# 3.5 Check if cluster could create life outlook difference
# 3.5.1 Subset life outlook score (OOL)
ool_data=data_clean['OOL']

# 3.5.2 Split OOL data into train and test sets 
# split with same random_state -> same dataset!
ool_train, ool_test = train_test_split(ool_data, test_size=.3, random_state=123)

# 3.5.3 Assign cluster to training set through merge()
ool_train1=pd.DataFrame(ool_train)
ool_train1.reset_index(level=0, inplace=True)
merged_train_all=pd.merge(ool_train1, merged_train, on='index')
sub1 = merged_train_all[['OOL', 'cluster']].dropna()

# 3.5.4 Calculate cluster statistics
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi 

# 1) Check for cluster difference (difference)
oolmod = smf.ols(formula='OOL ~ C(cluster)', data=sub1).fit()
print (oolmod.summary()) #p-value < 0.05 -> significant

# 2) Cluster mean (Cluster 2 > Cluster 0 > Cluster 1)
print ('means for OOL by cluster')
m1= sub1.groupby('cluster').mean()
print (m1)

# 3) Cluster sd
print ('standard deviations for GPA by cluster')
m2= sub1.groupby('cluster').std()
print (m2)

# 4) Compare cluster pair difference (all differenct)
mc1 = multi.MultiComparison(sub1['OOL'], sub1['cluster'])
res1 = mc1.tukeyhsd()
print(res1.summary()) 