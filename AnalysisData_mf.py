# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 21:29:09 2022

@author: Sneha Ray
"""

# In[134]:


import numpy as np
import pathlib
import os
import glob
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import pandas as pd
import seaborn as sns
# In[135]:

#subjectdir_above30 = "C:/Users/bkbme/Desktop/Sneha_program/data_analysis/age/above30/"
#subjectdir_upto30 = "C:/Users/bkbme/Desktop/Sneha_program/data_analysis/age/upto30/"
#subjectdir_upto21 = "C:/Users/bkbme/Desktop/Sneha_program/data_analysis/age/upto21/"



subjectdir_male = "C:/Users/bkbme/Desktop/Sneha_program/data_analysis1/gender/male/"
subjectdir_female = "C:/Users/bkbme/Desktop/Sneha_program/data_analysis1/gender/female/"

# In[136]:
allDir_male = glob.glob(subjectdir_male + os.sep + "*" + os.sep)
allDir_female = glob.glob(subjectdir_female + os.sep + "*" + os.sep)

#%% fetch the data for male subjects
allsubjectsdata_male = []
subject_num_male= 0
for parentdir in allDir_male:
    allTrialsSubject_male = glob.glob(parentdir + "*.csv")
    for trsubj in allTrialsSubject_male:
        try:
            data_male = np.genfromtxt(trsubj, delimiter=';')
            data_male = data_male[:, :5]
            data_male = np.column_stack((data_male, np.ones((data_male.shape[0], 1))*subject_num_male))
            allsubjectsdata_male.append(np.array(data_male))
            print(parentdir)
            print(subject_num_male)
        except:
            print("BAD " + str(trsubj))
    subject_num_male += 1

# In[137]: stack the data for all male subjects, all trials

allsubjectsdata_male = np.row_stack(allsubjectsdata_male)

#%% fetch the data for female subjects
allsubjectsdata_female = []
subject_num_female = 0
for parentdir in allDir_female:
    allTrialsSubject_female = glob.glob(parentdir + "*.csv")
    for trsubj in allTrialsSubject_female:
        try:
            data_female = np.genfromtxt(trsubj, delimiter=';')
            data_female = data_female[:, :5]
            data_female = np.column_stack((data_female, np.ones((data_female.shape[0], 1))*subject_num_female))
            allsubjectsdata_female.append(np.array(data_female))
            print(parentdir)
            print(subject_num_female)
        except:
            print("BAD " + str(trsubj))
    subject_num_female += 1

# In[137]:

allsubjectsdata_female = np.row_stack(allsubjectsdata_female)



# In[138]:


print(allsubjectsdata_male.shape)
print(allsubjectsdata_female.shape)


#%% Plot Histogram for male group
allsubjects_Latency_mean_male = np.zeros(np.unique(allsubjectsdata_male[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_median_male = np.zeros(np.unique(allsubjectsdata_male[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_stdev_male = np.zeros(np.unique(allsubjectsdata_male[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_min_male = np.zeros(np.unique(allsubjectsdata_male[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_max_male = np.zeros(np.unique(allsubjectsdata_male[:, -1]).shape, dtype=float, order='C')

allsubjects_Latency_target_male = np.zeros((18,12), dtype=float, order='C') # manually specify the no of subject and trials

targets_male = np.unique(allsubjectsdata_male[:, 4])

for s_num in np.unique(allsubjectsdata_male[:, -1]):
    try:
        good_index_male = allsubjectsdata_male[:, -1] == s_num
        good_data_male = allsubjectsdata_male[good_index_male, :]
        good_index_male = np.isfinite(good_data_male[:, 2] - good_data_male[:, 1])
        good_data_male = good_data_male[good_index_male, :]
        # remove invalid entries i.e under 100ms
        good_data_male = good_data_male[good_data_male[:, 2] >= 10000]
        allsubjects_Latency_mean_male[int(s_num)] = np.mean(good_data_male[:, 2])
        allsubjects_Latency_median_male[int(s_num)] = np.median(good_data_male[:, 2])
        allsubjects_Latency_stdev_male[int(s_num)] = np.std(good_data_male[:, 2])
        allsubjects_Latency_min_male[int(s_num)] = np.min(good_data_male[:, 2])
        allsubjects_Latency_max_male[int(s_num)] = np.max(good_data_male[:, 2])
        # compute mean latency per target per subject
        for target in targets_male:
            target_data_male = good_data_male[good_data_male[:, 4] == target]
            mean_target_male = np.round(np.mean(target_data_male[:, 2]))
            allsubjects_Latency_target_male[int(s_num),(int(target)-1)]=mean_target_male
            print("target: " + str(int(target)) + ", mean latency: " + str(mean_target_male))
        print("=============================")
        hist, bin_edges = np.histogram(good_data_male[:, 2]/1e6, 15)
        plt.plot(bin_edges[:-1], hist) #, label=str(good_data_male[0, -1]))
    except:
        print(s_num)
plt.legend()
plt.title('Latency graph of every Male subject')
plt.xlabel('Latency values in sec')
plt.ylabel('No. of Trials')
plt.show()

#%% Plot Histogram for female group
allsubjects_Latency_mean_female = np.zeros(np.unique(allsubjectsdata_female[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_median_female = np.zeros(np.unique(allsubjectsdata_female[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_stdev_female = np.zeros(np.unique(allsubjectsdata_female[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_min_female = np.zeros(np.unique(allsubjectsdata_female[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_max_female = np.zeros(np.unique(allsubjectsdata_female[:, -1]).shape, dtype=float, order='C')

allsubjects_Latency_target_female = np.zeros((21,12), dtype=float, order='C') # manually specify the no of subject and trials

targets_female = np.unique(allsubjectsdata_female[:, 4])
#%
for s_num in np.unique(allsubjectsdata_female[:, -1]):
    try:
        good_index_female = allsubjectsdata_female[:, -1] == s_num
        good_data_female = allsubjectsdata_female[good_index_female, :]
        good_index_female = np.isfinite(good_data_female[:, 2] - good_data_female[:, 1])
        good_data_female = good_data_female[good_index_female, :]
        # remove invalid entries i.e under 100ms
        good_data_female = good_data_female[good_data_female[:, 2] >= 10000]
        allsubjects_Latency_mean_female[int(s_num)] = np.mean(good_data_female[:, 2])
        allsubjects_Latency_median_female[int(s_num)] = np.median(good_data_female[:, 2])
        allsubjects_Latency_stdev_female[int(s_num)] = np.std(good_data_female[:, 2])
        allsubjects_Latency_min_female[int(s_num)] = np.min(good_data_female[:, 2])
        allsubjects_Latency_max_female[int(s_num)] = np.max(good_data_female[:, 2])
        # compute mean latency per target per subject
        for target in targets_female:
            target_data_female = good_data_female[good_data_female[:, 4] == target]
            mean_target_female = np.round(np.mean(target_data_female[:, 2]))
            allsubjects_Latency_target_female[int(s_num),(int(target)-1)]=mean_target_female
            print("target: " + str(int(target)) + ", mean latency: " + str(mean_target_female))
        print("=============================")
        hist, bin_edges_female = np.histogram(good_data_female[:, 2]/1e6, 15)
        plt.plot(bin_edges[:-1], hist)#, label=str(good_data_female[0, -1]))
    except:
        print(s_num)
plt.legend()
plt.title('Latency graph of every female subject')
plt.xlabel('Latency values in sec')
plt.ylabel('No. of Trials')
plt.show()

#%% histogram of two group
bins=20
allsubjectsdata_latency_female = allsubjectsdata_female[:,2]
plt.figure(figsize=(12,8))
plt.hist(allsubjectsdata_latency_female,bins)
plt.title('Hitsogram for Latency of Female', fontsize=35)
plt.ylabel('No of Trial')
plt.xlabel('Latency values')
#plt.show()
allsubjectsdata_latency_male = allsubjectsdata_male[:,2]
plt.hist(allsubjectsdata_latency_male,bins)
plt.title('Hitsogram for Latency of male', fontsize=35)
plt.title('Hitsogram for all trials latency', fontsize=35)
plt.ylabel('No of Trial', fontsize=22)
plt.xlabel('Latency values', fontsize=22)
plt.show()
#%% violin plot for latency
allsubjects_Latency_mean_male_n = allsubjects_Latency_mean_male;
for x in range(2):
    allsubjects_Latency_mean_male_n = np.append(allsubjects_Latency_mean_male_n, [np.nan])
Latency_mean_df = pd.DataFrame({'Male_Mean_Latency': allsubjects_Latency_mean_male_n, 'Female_Mean_Latency': allsubjects_Latency_mean_female})
plt.figure(figsize=(12,8))
my_palette = {'Male_Mean_Latency':'g', 'Female_Mean_Latency':'b'}
ax = sns.violinplot(data=Latency_mean_df, palette=my_palette, linewidth=3, saturation = 0.5)
ax = sns.swarmplot(data=Latency_mean_df, edgecolor="gray", size=22, alpha=0.9)
plt.rc('xtick', labelsize=25); plt.rc('ytick', labelsize=25)
plt.ylabel('Mean Latency', fontsize=22)
plt.title('Mean Latency', fontsize=35)  
plt.show()
#%% Mean,median and standard deviation of latency for male and female group and statistics between group
allsubjectsdata_latency_male_mean = np.mean(allsubjects_Latency_mean_male)
print('Mean of Latency for male group = ' + str(allsubjectsdata_latency_male_mean))

allsubjectsdata_latency_female_mean = np.mean(allsubjects_Latency_mean_female)
print('Mean of Latency for female group = ' + str(allsubjectsdata_latency_female_mean))

allsubjectsdata_latency_male_stdev = np.mean(allsubjects_Latency_stdev_male)
print('Stdev of Latency for male group = ' + str(allsubjectsdata_latency_male_stdev))

allsubjectsdata_latency_female_stdev = np.mean(allsubjects_Latency_stdev_female)
print('Stdev of Latency for female group = ' + str(allsubjectsdata_latency_female_stdev))

allsubjectsdata_latency_male_median = np.median(allsubjects_Latency_median_male)
print('Median of Latency for male group = ' + str(allsubjectsdata_latency_male_median))

allsubjectsdata_latency_female_median = np.median(allsubjects_Latency_median_female)
print('Median of Latency for female group = ' + str(allsubjectsdata_latency_female_median))
#%% calculating statistics (t and p values) between two groups
#t, p = ttest_ind(allsubjectsdata_male, allsubjectsdata_female, equal_var=False)
#print('p value between male vs female group = ' + str(p[2]))
#print('t value between male vs female group = ' + str(t[2]))
t, p = ttest_ind(allsubjects_Latency_mean_male, allsubjects_Latency_mean_female, equal_var=False)
print('p value between male vs female group = ' + str(p))
print('t value between male vs female group = ' + str(t))

#%%
t, p = ttest_ind(allsubjects_Latency_target_female, allsubjects_Latency_target_male, equal_var=False)
print('p value between male vs female group = ' + str(p))
print('t value between male vs female group = ' + str(t))

#%% violin plot for latency
tn=8
allsubjects_Latency_target_male_s=allsubjects_Latency_target_male[:,tn]
allsubjects_Latency_target_female_s=allsubjects_Latency_target_female[:,tn]
allsubjects_Latency_target_male_s_n = allsubjects_Latency_target_male_s;
for x in range(2):
    allsubjects_Latency_target_male_s_n = np.append(allsubjects_Latency_target_male_s_n, [np.nan])
Latency_mean_s_df = pd.DataFrame({'Male_Latency': allsubjects_Latency_target_male_s_n, 'Female_Latency': allsubjects_Latency_target_female_s})
plt.figure(figsize=(12,8))
my_palette = {'Male_Latency':'g', 'Female_Latency':'b'}
ax = sns.violinplot(data=Latency_mean_s_df, palette=my_palette, linewidth=3, saturation = 0.5)
ax = sns.swarmplot(data=Latency_mean_s_df, edgecolor="gray", size=12, alpha=0.9)
plt.rc('xtick', labelsize=25); plt.rc('ytick', labelsize=25)
plt.ylabel('Mean Latency', fontsize=22)
plt.title('Mean Latency', fontsize=35)  
plt.show()
