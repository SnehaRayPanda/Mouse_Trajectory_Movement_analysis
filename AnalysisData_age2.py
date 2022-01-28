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

subjectdir_old = "C:/Users/bkbme/Desktop/Sneha_program/data_analysis1/age/above30/"
subjectdir_Adult = "C:/Users/bkbme/Desktop/Sneha_program/data_analysis1/age/upto30/"
subjectdir_young = "C:/Users/bkbme/Desktop/Sneha_program/data_analysis1/age/upto21/"
# In[136]:
allDir_old = glob.glob(subjectdir_old + os.sep + "*" + os.sep)
allDir_Adult = glob.glob(subjectdir_Adult + os.sep + "*" + os.sep)
allDir_young = glob.glob(subjectdir_young + os.sep + "*" + os.sep)
#%% fetch the data for old subjects
allsubjectsdata_old = []
subject_num_old= 0
for parentdir in allDir_old:
    allTrialsSubject_old = glob.glob(parentdir + "*.csv")
    for trsubj in allTrialsSubject_old:
        try:
            data_old = np.genfromtxt(trsubj, delimiter=';')
            data_old = data_old[:, :5]
            data_old = np.column_stack((data_old, np.ones((data_old.shape[0], 1))*subject_num_old))
            allsubjectsdata_old.append(np.array(data_old))
            print(parentdir)
            print(subject_num_old)
        except:
            print("BAD " + str(trsubj))
    subject_num_old += 1

# In[137]: stack the data for all old subjects, all trials

allsubjectsdata_old = np.row_stack(allsubjectsdata_old)

#%% fetch the data for Adult subjects
allsubjectsdata_Adult = []
subject_num_Adult = 0
for parentdir in allDir_Adult:
    allTrialsSubject_Adult = glob.glob(parentdir + "*.csv")
    for trsubj in allTrialsSubject_Adult:
        try:
            data_Adult = np.genfromtxt(trsubj, delimiter=';')
            data_Adult = data_Adult[:, :5]
            data_Adult = np.column_stack((data_Adult, np.ones((data_Adult.shape[0], 1))*subject_num_Adult))
            allsubjectsdata_Adult.append(np.array(data_Adult))
            print(parentdir)
            print(subject_num_Adult)
        except:
            print("BAD " + str(trsubj))
    subject_num_Adult += 1

# In[137]:

allsubjectsdata_Adult = np.row_stack(allsubjectsdata_Adult)
#%% fetch the data for young subjects
allsubjectsdata_young = []
subject_num_young = 0
for parentdir in allDir_young:
    allTrialsSubject_young = glob.glob(parentdir + "*.csv")
    for trsubj in allTrialsSubject_young:
        try:
            data_young = np.genfromtxt(trsubj, delimiter=';')
            data_young = data_young[:, :5]
            data_young = np.column_stack((data_young, np.ones((data_young.shape[0], 1))*subject_num_young))
            allsubjectsdata_young.append(np.array(data_young))
            print(parentdir)
            print(subject_num_young)
        except:
            print("BAD " + str(trsubj))
    subject_num_young += 1

# In[137]:

allsubjectsdata_young = np.row_stack(allsubjectsdata_young)

# In[138]:


print(allsubjectsdata_old.shape)
print(allsubjectsdata_Adult.shape)
print(allsubjectsdata_young.shape)

#%% Plot Histogram for old group
allsubjects_Latency_mean_old = np.zeros(np.unique(allsubjectsdata_old[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_median_old = np.zeros(np.unique(allsubjectsdata_old[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_stdev_old = np.zeros(np.unique(allsubjectsdata_old[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_min_old = np.zeros(np.unique(allsubjectsdata_old[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_max_old = np.zeros(np.unique(allsubjectsdata_old[:, -1]).shape, dtype=float, order='C')

allsubjects_Latency_target_old = np.zeros((8,12), dtype=float, order='C') # manually specify the no of subject and trials

targets_old = np.unique(allsubjectsdata_old[:, 4])

for s_num in np.unique(allsubjectsdata_old[:, -1]):
    try:
        good_index_old = allsubjectsdata_old[:, -1] == s_num
        good_data_old = allsubjectsdata_old[good_index_old, :]
        good_index_old = np.isfinite(good_data_old[:, 2] - good_data_old[:, 1])
        good_data_old = good_data_old[good_index_old, :]
        # remove invalid entries i.e under 100ms
        good_data_old = good_data_old[good_data_old[:, 2] >= 10000]
        allsubjects_Latency_mean_old[int(s_num)] = np.mean(good_data_old[:, 2])
        print("Subject: " + str(int(s_num)) + ", mean latency: " + str(allsubjects_Latency_mean_old[int(s_num)]))
        allsubjects_Latency_median_old[int(s_num)] = np.median(good_data_old[:, 2])
        allsubjects_Latency_stdev_old[int(s_num)] = np.std(good_data_old[:, 2])
        allsubjects_Latency_min_old[int(s_num)] = np.min(good_data_old[:, 2])
        allsubjects_Latency_max_old[int(s_num)] = np.max(good_data_old[:, 2])
        # compute mean latency per target per subject
        for target in targets_old:
            target_data_old = good_data_old[good_data_old[:, 4] == target]
            mean_target_old = np.round(np.mean(target_data_old[:, 2]))
            allsubjects_Latency_target_old[int(s_num),(int(target)-1)]=mean_target_old
            print("target: " + str(int(target)) + ", mean latency: " + str(mean_target_old))
        print("=============================")
        hist, bin_edges = np.histogram(good_data_old[:, 2]/1e6, 15)
        plt.plot(bin_edges[:-1], hist)#, label=str(good_data_old[0, -1]))
    except:
        print(s_num)
plt.legend()
plt.title('Latency graph of every old subject')
plt.xlabel('Latency values in sec')
plt.ylabel('No. of Trials')
plt.show()

#%% Plot Histogram for Adult group
allsubjects_Latency_mean_Adult = np.zeros(np.unique(allsubjectsdata_Adult[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_median_Adult = np.zeros(np.unique(allsubjectsdata_Adult[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_stdev_Adult = np.zeros(np.unique(allsubjectsdata_Adult[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_min_Adult = np.zeros(np.unique(allsubjectsdata_Adult[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_max_Adult = np.zeros(np.unique(allsubjectsdata_Adult[:, -1]).shape, dtype=float, order='C')

allsubjects_Latency_target_Adult = np.zeros((17,12), dtype=float, order='C') # manually specify the no of subject and trials

targets_Adult = np.unique(allsubjectsdata_Adult[:, 4])
#%
for s_num in np.unique(allsubjectsdata_Adult[:, -1]):
    try:
        good_index_Adult = allsubjectsdata_Adult[:, -1] == s_num
        good_data_Adult = allsubjectsdata_Adult[good_index_Adult, :]
        good_index_Adult = np.isfinite(good_data_Adult[:, 2] - good_data_Adult[:, 1])
        good_data_Adult = good_data_Adult[good_index_Adult, :]
        # remove invalid entries i.e under 100ms
        good_data_Adult = good_data_Adult[good_data_Adult[:, 2] >= 10000]
        allsubjects_Latency_mean_Adult[int(s_num)] = np.mean(good_data_Adult[:, 2])
        print("Subject: " + str(int(s_num)) + ", mean latency: " + str(allsubjects_Latency_mean_Adult[int(s_num)]))
        allsubjects_Latency_median_Adult[int(s_num)] = np.median(good_data_Adult[:, 2])
        allsubjects_Latency_stdev_Adult[int(s_num)] = np.std(good_data_Adult[:, 2])
        allsubjects_Latency_min_Adult[int(s_num)] = np.min(good_data_Adult[:, 2])
        allsubjects_Latency_max_Adult[int(s_num)] = np.max(good_data_Adult[:, 2])
        # compute mean latency per target per subject
        for target in targets_Adult:
            target_data_Adult = good_data_Adult[good_data_Adult[:, 4] == target]
            mean_target_Adult = np.round(np.mean(target_data_Adult[:, 2]))
            allsubjects_Latency_target_Adult[int(s_num),(int(target)-1)]=mean_target_Adult
            print("target: " + str(int(target)) + ", mean latency: " + str(mean_target_Adult))
        print("=============================")
        hist, bin_edges_Adult = np.histogram(good_data_Adult[:, 2]/1e6, 15)
        plt.plot(bin_edges[:-1], hist)#, label=str(good_data_Adult[0, -1]))
    except:
        print(s_num)
plt.legend()
plt.title('Latency graph of every Adult subject')
plt.xlabel('Latency values in sec')
plt.ylabel('No. of Trials')
plt.show()



#%% Plot Histogram for young group
allsubjects_Latency_mean_young = np.zeros(np.unique(allsubjectsdata_young[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_median_young = np.zeros(np.unique(allsubjectsdata_young[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_stdev_young = np.zeros(np.unique(allsubjectsdata_young[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_min_young = np.zeros(np.unique(allsubjectsdata_young[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_max_young = np.zeros(np.unique(allsubjectsdata_young[:, -1]).shape, dtype=float, order='C')

allsubjects_Latency_target_young = np.zeros((11,12), dtype=float, order='C') # manually specify the no of subject and trials

targets_young = np.unique(allsubjectsdata_young[:, 4])
#%
for s_num in np.unique(allsubjectsdata_young[:, -1]):
    try:
        good_index_young = allsubjectsdata_young[:, -1] == s_num
        good_data_young = allsubjectsdata_young[good_index_young, :]
        good_index_young = np.isfinite(good_data_young[:, 2] - good_data_young[:, 1])
        good_data_young = good_data_young[good_index_young, :]
        # remove invalid entries i.e under 100ms
        good_data_young = good_data_young[good_data_young[:, 2] >= 10000]
        allsubjects_Latency_mean_young[int(s_num)] = np.mean(good_data_young[:, 2])
        print("Subject: " + str(int(s_num)) + ", mean latency: " + str(allsubjects_Latency_mean_young[int(s_num)]))
        allsubjects_Latency_median_young[int(s_num)] = np.median(good_data_young[:, 2])
        allsubjects_Latency_stdev_young[int(s_num)] = np.std(good_data_young[:, 2])
        allsubjects_Latency_min_young[int(s_num)] = np.min(good_data_young[:, 2])
        allsubjects_Latency_max_young[int(s_num)] = np.max(good_data_young[:, 2])
        # compute mean latency per target per subject
        for target in targets_young:
            target_data_young = good_data_young[good_data_young[:, 4] == target]
            mean_target_young = np.round(np.mean(target_data_young[:, 2]))
            allsubjects_Latency_target_young[int(s_num),(int(target)-1)]=mean_target_young
            print("target: " + str(int(target)) + ", mean latency: " + str(mean_target_young))
        print("=============================")
        hist, bin_edges_young = np.histogram(good_data_young[:, 2]/1e6, 15)
        plt.plot(bin_edges[:-1], hist, label=str(good_data_young[0, -1]))
    except:
        print(s_num)
plt.legend()
plt.title('Latency graph of every young subject')
plt.xlabel('Latency values in sec')
plt.ylabel('No. of Trials')
plt.show()
#%% histogram of two group
bins=20
plt.figure(figsize=(12,8))
allsubjectsdata_latency_Adult = allsubjectsdata_Adult[:,2]
plt.hist(allsubjectsdata_latency_Adult,bins)
plt.title('Hitsogram Representation for Latency values of Adult')
plt.ylabel('No of Trial')
plt.xlabel('Latency values')
#plt.show()
allsubjectsdata_latency_young = allsubjectsdata_young[:,2]
plt.hist(allsubjectsdata_latency_young,bins)
plt.title('Hitsogram Representation for Latency values of Young')
plt.ylabel('No of Trial')
plt.xlabel('Latency values')
#plt.show()
#%
allsubjectsdata_latency_old = allsubjectsdata_old[:,2]
plt.hist(allsubjectsdata_latency_old,bins)
plt.title('Hitsogram Representation for Latency values of old')
plt.title('Hitsogram Representation for Latency values', fontsize=35)
plt.ylabel('No of Trial', fontsize=22)
plt.xlabel('Latency, Young=Orange, Adult=Blue, Old=Green', fontsize=22)
plt.show()
#%% violin plot for mean latency
allsubjects_Latency_mean_old_n = allsubjects_Latency_mean_old;
for x in range(9):
    allsubjects_Latency_mean_old_n = np.append(allsubjects_Latency_mean_old_n, [np.nan])
    
allsubjects_Latency_mean_young_n = allsubjects_Latency_mean_young;
for x in range(6):
    allsubjects_Latency_mean_young_n = np.append(allsubjects_Latency_mean_young_n, [np.nan])
    
Latency_mean_df = pd.DataFrame({'Old_Latency': allsubjects_Latency_mean_old_n, 'Adult_Latency': allsubjects_Latency_mean_Adult, 'Young_Latency': allsubjects_Latency_mean_young_n})
plt.figure(figsize=(12,8))
my_palette = {'Old_Latency':'g', 'Adult_Latency':'b', 'Young_Latency':'r'}
ax = sns.violinplot(data=Latency_mean_df, palette=my_palette, linewidth=3, saturation = 0.5)
ax = sns.swarmplot(data=Latency_mean_df, edgecolor="gray", size=12, alpha=0.9)
plt.rc('xtick', labelsize=25); plt.rc('ytick', labelsize=22)
plt.ylabel('Mean Latency', fontsize=22)
plt.title('Mean Latency', fontsize=35)  
plt.ylim(300000,800000)
plt.show()

#%% violin plot for median latency
allsubjects_Latency_median_old_n = allsubjects_Latency_median_old;
for x in range(9):
    allsubjects_Latency_median_old_n = np.append(allsubjects_Latency_median_old_n, [np.nan])
    
allsubjects_Latency_median_young_n = allsubjects_Latency_median_young;
for x in range(6):
    allsubjects_Latency_median_young_n = np.append(allsubjects_Latency_median_young_n, [np.nan])
    
Latency_median_df = pd.DataFrame({'Old_Latency': allsubjects_Latency_median_old_n, 'Adult_Latency': allsubjects_Latency_median_Adult, 'Young_Latency': allsubjects_Latency_median_young_n})
plt.figure(figsize=(12,8))
my_palette = {'Old_Latency':'g', 'Adult_Latency':'b', 'Young_Latency':'r'}
ax = sns.violinplot(data=Latency_median_df, palette=my_palette, linewidth=3, saturation = 0.5)
ax = sns.swarmplot(data=Latency_median_df, edgecolor="gray", size=12, alpha=0.9)
plt.rc('xtick', labelsize=25); plt.rc('ytick', labelsize=25)
plt.ylabel('median Latency', fontsize=22)
plt.title('Median Latency', fontsize=35)  
plt.show()
#%% Mean,median and standard deviation of latency for old and Adult group and statistics between group
allsubjectsdata_latency_old_mean = np.mean(allsubjects_Latency_mean_old)
print('Mean of Latency for old group = ' + str(allsubjectsdata_latency_old_mean))

allsubjectsdata_latency_Adult_mean = np.mean(allsubjects_Latency_mean_Adult)
print('Mean of Latency for Adult group = ' + str(allsubjectsdata_latency_Adult_mean))

allsubjectsdata_latency_young_mean = np.mean(allsubjects_Latency_mean_young)
print('Mean of Latency for Young group = ' + str(allsubjectsdata_latency_young_mean))

allsubjectsdata_latency_old_stdev = np.mean(allsubjects_Latency_stdev_old)
print('Stdev of Latency for old group = ' + str(allsubjectsdata_latency_old_stdev))

allsubjectsdata_latency_Adult_stdev = np.mean(allsubjects_Latency_stdev_Adult)
print('Stdev of Latency for Adult group = ' + str(allsubjectsdata_latency_Adult_stdev))

allsubjectsdata_latency_young_stdev = np.mean(allsubjects_Latency_stdev_young)
print('Stdev of Latency for young group = ' + str(allsubjectsdata_latency_young_stdev))

allsubjectsdata_latency_old_median = np.median(allsubjects_Latency_median_old)
print('Median of Latency for old group = ' + str(allsubjectsdata_latency_old_median))

allsubjectsdata_latency_Adult_median = np.median(allsubjects_Latency_median_Adult)
print('Median of Latency for Adult group = ' + str(allsubjectsdata_latency_Adult_median))
#%% calculating statistics (t and p values) between two groups
t, p = ttest_ind(allsubjects_Latency_median_old, allsubjects_Latency_median_Adult, equal_var=False)
print('p value between old vs Adult group = ' + str(p))
print('t value between old vs Adult group = ' + str(t))
print('')
t, p = ttest_ind(allsubjects_Latency_median_old, allsubjects_Latency_median_young, equal_var=False)
print('p value between old vs Young group = ' + str(p))
print('t value between old vs Young group = ' + str(t))
print('')
t, p = ttest_ind(allsubjects_Latency_median_Adult, allsubjects_Latency_median_young, equal_var=False)
print('p value between Adult vs young group = ' + str(p))
print('t value between Adult vs young group = ' + str(t))
#%%
t, p = ttest_ind(allsubjects_Latency_target_Adult, allsubjects_Latency_target_old, equal_var=False)
print('p value between old vs Adult group = ' + str(p))
print('t value between old vs Adult group = ' + str(t))
print('')
t, p = ttest_ind(allsubjects_Latency_target_young, allsubjects_Latency_target_Adult, equal_var=False)
print('p value between young vs Adult group = ' + str(p))
print('t value between young vs Adult group = ' + str(t))
print('')
t, p = ttest_ind(allsubjects_Latency_target_young, allsubjects_Latency_target_old, equal_var=False)
print('p value between young vs Old group = ' + str(p))
print('t value between young vs Old group = ' + str(t))
#%% violin plot for latency
tn=4 # specifay which target you want to plot
allsubjects_Latency_target_old_s=allsubjects_Latency_target_old[:,tn]
allsubjects_Latency_target_Adult_s=allsubjects_Latency_target_Adult[:,tn]
allsubjects_Latency_target_young_s=allsubjects_Latency_target_young[:,tn]

allsubjects_Latency_target_old_s_n = allsubjects_Latency_target_old_s;
for x in range(9):
    allsubjects_Latency_target_old_s_n = np.append(allsubjects_Latency_target_old_s_n, [np.nan])

allsubjects_Latency_target_young_s_n = allsubjects_Latency_target_young_s;
for x in range(6):
    allsubjects_Latency_target_young_s_n = np.append(allsubjects_Latency_target_young_s_n, [np.nan])
    
Latency_mean_s_df = pd.DataFrame({'Old_Latency': allsubjects_Latency_target_old_s_n, 'Adult_Latency': allsubjects_Latency_target_Adult_s, 'Young_Latency': allsubjects_Latency_target_young_s_n})
plt.figure(figsize=(12,8))
my_palette = {'Old_Latency':'g', 'Adult_Latency':'b', 'Young_Latency':'r'}
ax = sns.violinplot(data=Latency_mean_s_df, palette=my_palette, linewidth=3, saturation = 0.5)
ax = sns.swarmplot(data=Latency_mean_s_df, edgecolor="gray", size=12, alpha=0.9)
plt.rc('xtick', labelsize=25); plt.rc('ytick', labelsize=25)
plt.ylabel('Mean Latency of target 5', fontsize=22) # change target name also
plt.title('Mean Latency of target 5', fontsize=35)  # change target name also
plt.show()