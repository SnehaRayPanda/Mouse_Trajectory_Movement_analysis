# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 16:08:43 2022

@author: Sneha Ray
"""

import numpy as np
import pathlib
import os
import glob
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import scipy.stats as stats
import pandas as pd 

# In[131]:


subjectdir_above30 = "C:/Users/bkbme/Desktop/Sneha_program/data_analysis/age/above30/"
subjectdir_upto30 = "C:/Users/bkbme/Desktop/Sneha_program/data_analysis/age/upto30/"
subjectdir_upto21 = "C:/Users/bkbme/Desktop/Sneha_program/data_analysis/age/upto21/"



# In[136]:


allDir_above30 = glob.glob(subjectdir_above30 + os.sep + "*" + os.sep)
allDir_upto30 = glob.glob(subjectdir_upto30 + os.sep + "*" + os.sep)
allDir_upto21 = glob.glob(subjectdir_upto21 + os.sep + "*" + os.sep)

#%% to extract all the subject data above 25
allsubjectsdata_above30 = []
subject_num_above30 = 0
for parentdir in allDir_above30:
    allTrialsSubject_above30 = glob.glob(parentdir + "*.csv")
    for trsubj in allTrialsSubject_above30:
        try:
            data_above30 = np.genfromtxt(trsubj, delimiter=';')
            data_above30 = data_above30[:, :5]
            data_above30 = np.column_stack((data_above30, np.ones((data_above30.shape[0], 1))*subject_num_above30))
            allsubjectsdata_above30.append(np.array(data_above30))
            print(parentdir)
            print(subject_num_above30)
        except:
            print("BAD " + str(trsubj))
    subject_num_above30 += 1

# In[137]:
allsubjectsdata_above30 = np.row_stack(allsubjectsdata_above30)


#%% to extract all the subject data upto 25
allsubjectsdata_upto30 = []
subject_num_upto30 = 0
for parentdir in allDir_upto30:
    allTrialsSubject_upto30 = glob.glob(parentdir + "*.csv")
    for trsubj in allTrialsSubject_upto30:
        try:
            data_upto30 = np.genfromtxt(trsubj, delimiter=';')
            data_upto30 = data_upto30[:, :5]
            data_upto30 = np.column_stack((data_upto30, np.ones((data_upto30.shape[0], 1))*subject_num_upto30))
            allsubjectsdata_upto30.append(np.array(data_upto30))
            print(parentdir)
            print(subject_num_upto30)
        except:
            print("BAD " + str(trsubj))
    subject_num_upto30 += 1
# In[137]:
allsubjectsdata_upto30 = np.row_stack(allsubjectsdata_upto30)


#%% to extract all the subject data upto 21
allsubjectsdata_upto21 = []
subject_num_upto21 = 0
for parentdir in allDir_upto21:
    allTrialsSubject_upto21 = glob.glob(parentdir + "*.csv")
    for trsubj in allTrialsSubject_upto21:
        try:
            data_upto21 = np.genfromtxt(trsubj, delimiter=';')
            data_upto21 = data_upto21[:, :5]
            data_upto21 = np.column_stack((data_upto21, np.ones((data_upto21.shape[0], 1))*subject_num_upto21))
            allsubjectsdata_upto21.append(np.array(data_upto21))
            print(parentdir)
            print(subject_num_upto21)
        except:
            print("BAD " + str(trsubj))
    subject_num_upto21 += 1
# In[137]:
allsubjectsdata_upto21 = np.row_stack(allsubjectsdata_upto21)


#%% to plot latency graph for every subjects above 25
allsubjects_Latency_mean_above30 = np.zeros(np.unique(allsubjectsdata_above30[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_median_above30 = np.zeros(np.unique(allsubjectsdata_above30[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_stdev_above30 = np.zeros(np.unique(allsubjectsdata_above30[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_min_above30 = np.zeros(np.unique(allsubjectsdata_above30[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_max_above30 = np.zeros(np.unique(allsubjectsdata_above30[:, -1]).shape, dtype=float, order='C')

for s_num in np.unique(allsubjectsdata_above30[:, -1]):
    try:
        good_index_above30 = allsubjectsdata_above30[:, -1] == s_num
        good_data_above30 = allsubjectsdata_above30[good_index_above30, :]
        good_index_above30 = np.isfinite(good_data_above30[:, 2] - good_data_above30[:, 1])
        allsubjects_Latency_mean_above30[int(s_num)] = np.mean(good_data_above30[:, 2])
        allsubjects_Latency_median_above30[int(s_num)] = np.median(good_data_above30[:, 2])
        allsubjects_Latency_stdev_above30[int(s_num)] = np.std(good_data_above30[:, 2])
        allsubjects_Latency_min_above30[int(s_num)] = np.min(good_data_above30[:, 2])
        allsubjects_Latency_max_above30[int(s_num)] = np.max(good_data_above30[:, 2])
        good_data_above30 = good_data_above30[good_index_above30, :]
        hist, bin_edges = np.histogram(good_data_above30[:, 2]/1e6, 15)
        plt.plot(bin_edges[:-1], hist)#, label=str(good_data_above30[0, -1]))
    except:
        print(s_num)
plt.legend()
plt.title('Latency graph of subjects above 30yr')
plt.xlabel('Latency values in sec')
plt.ylabel('No. of Trials')
plt.show()

#%% to plot latency graph for every subjects upto 25
allsubjects_Latency_mean_upto30 = np.zeros(np.unique(allsubjectsdata_upto30[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_median_upto30 = np.zeros(np.unique(allsubjectsdata_upto30[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_stdev_upto30 = np.zeros(np.unique(allsubjectsdata_upto30[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_min_upto30 = np.zeros(np.unique(allsubjectsdata_upto30[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_max_upto30 = np.zeros(np.unique(allsubjectsdata_upto30[:, -1]).shape, dtype=float, order='C')

for s_num in np.unique(allsubjectsdata_upto30[:, -1]):
    try:
        good_index_upto30 = allsubjectsdata_upto30[:, -1] == s_num
        good_data_upto30 = allsubjectsdata_upto30[good_index_upto30, :]
        good_index_upto30 = np.isfinite(good_data_upto30[:, 2] - good_data_upto30[:, 1])
        allsubjects_Latency_mean_upto30[int(s_num)] = np.mean(good_data_upto30[:, 2])
        allsubjects_Latency_median_upto30[int(s_num)] = np.median(good_data_upto30[:, 2])
        allsubjects_Latency_stdev_upto30[int(s_num)] = np.std(good_data_upto30[:, 2])
        allsubjects_Latency_min_upto30[int(s_num)] = np.min(good_data_upto30[:, 2])
        allsubjects_Latency_max_upto30[int(s_num)] = np.max(good_data_upto30[:, 2])
        good_data_upto30 = good_data_upto30[good_index_upto30, :]
        hist, bin_edges = np.histogram(good_data_upto30[:, 2]/1e6, 15)
        plt.plot(bin_edges[:-1], hist)#, label=str(good_data_upto30[0, -1]))
    except:
        print(s_num)
plt.legend()
plt.title('Latency graph of subjects upto 30yr')
plt.xlabel('Latency values in sec')
plt.ylabel('No. of Trials')
plt.show()

#%% to plot latency graph for every subjects upto 21
allsubjects_Latency_mean_upto21 = np.zeros(np.unique(allsubjectsdata_upto21[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_median_upto21 = np.zeros(np.unique(allsubjectsdata_upto21[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_stdev_upto21 = np.zeros(np.unique(allsubjectsdata_upto21[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_min_upto21 = np.zeros(np.unique(allsubjectsdata_upto21[:, -1]).shape, dtype=float, order='C')
allsubjects_Latency_max_upto21 = np.zeros(np.unique(allsubjectsdata_upto21[:, -1]).shape, dtype=float, order='C')

for s_num in np.unique(allsubjectsdata_upto21[:, -1]):
    try:
        good_index_upto21 = allsubjectsdata_upto21[:, -1] == s_num
        good_data_upto21 = allsubjectsdata_upto21[good_index_upto21, :]
        good_index_upto21 = np.isfinite(good_data_upto21[:, 2] - good_data_upto21[:, 1])
        allsubjects_Latency_mean_upto21[int(s_num)] = np.mean(good_data_upto21[:, 2])
        allsubjects_Latency_median_upto21[int(s_num)] = np.median(good_data_upto21[:, 2])
        allsubjects_Latency_stdev_upto21[int(s_num)] = np.std(good_data_upto21[:, 2])
        allsubjects_Latency_min_upto21[int(s_num)] = np.min(good_data_upto21[:, 2])
        allsubjects_Latency_max_upto21[int(s_num)] = np.max(good_data_upto21[:, 2])
        good_data_upto21 = good_data_upto21[good_index_upto21, :]
        hist, bin_edges = np.histogram(good_data_upto21[:, 2]/1e6, 15)
        plt.plot(bin_edges[:-1], hist)#, label=str(good_data_upto21[0, -1]))
    except:
        print(s_num)
plt.legend()
plt.title('Latency graph of subjects upto 21yr')
plt.xlabel('Latency values in sec')
plt.ylabel('No. of Trials')
plt.show()

#%% to plot histogram of every group latency
plt.hist(allsubjectsdata_above30[:,2])
plt.title('Hitsogram Representation for Latency values of Above 30yr')
plt.ylabel('No of Trial')
plt.xlabel('Latency values')
plt.show()
plt.hist(allsubjectsdata_upto30[:,2])
plt.title('Hitsogram Representation for Latency values of Upto 30yr')
plt.ylabel('No of Trial')
plt.xlabel('Latency values')
plt.show()
plt.hist(allsubjectsdata_upto21[:,2])
plt.title('Hitsogram Representation for Latency values of Upto 21yr')
plt.ylabel('No of Trial')
plt.xlabel('Latency values')
plt.show()

#%% Mean,median and standard deviation of latency for aging group and statistics between group
allsubjectsdata_above30_mean = np.mean(allsubjects_Latency_mean_above30)
print('Mean of Latency values of subjects above 30 yr= ' + str(allsubjectsdata_above30_mean))
allsubjectsdata_above30_median = np.median(allsubjects_Latency_median_above30)
print('Median of Latency values of subjects above 30 yr= ' + str(allsubjectsdata_above30_median))
allsubjectsdata_above30_stdev = np.mean(allsubjects_Latency_stdev_above30)
print('Stdev of Latency values of subjects above 30 yr= ' + str(allsubjectsdata_above30_stdev))

print('')

allsubjectsdata_upto30_mean = np.mean(allsubjects_Latency_mean_upto30)
print('Mean of Latency values of subjects 22-30 yr= ' + str(allsubjectsdata_upto30_mean))
allsubjectsdata_upto30_median = np.median(allsubjects_Latency_median_upto30)
print('Median of Latency values of subjects 22-30 yr= ' + str(allsubjectsdata_upto30_median))
allsubjectsdata_upto30_stdev = np.mean(allsubjects_Latency_stdev_upto30)
print('Stdev of Latency values of subjects 22-30 yr= ' + str(allsubjectsdata_upto30_stdev))

print(' ')

allsubjectsdata_upto21_mean = np.mean(allsubjects_Latency_mean_upto21)
print('Mean of Latency values of subjects upto 21 yr= ' + str(allsubjectsdata_upto21_mean))
allsubjectsdata_upto21_median = np.median(allsubjects_Latency_median_upto21)
print('Median of Latency values of subjects upto 21 yr= ' + str(allsubjectsdata_upto21_median))
allsubjectsdata_upto21_stdev = np.mean(allsubjects_Latency_stdev_upto21)
print('Stdev of Latency values of subjects upto 21 yr= ' + str(allsubjectsdata_upto21_stdev))
print(' ')


#%% calculate p,t values of every aging groups using simple t-test
t_Above30vsUpto30, p_Above30vsUpto30 = ttest_ind(allsubjectsdata_above30, allsubjectsdata_upto30, equal_var=False)
print('p value between Above30 and Upto30 group = ' + str(p_Above30vsUpto30[2]))
print('t value between Above30 and Upto30 group = ' + str(t_Above30vsUpto30[2]))
print(' ')
t_Above30vsUpto21, p_Above30vsUpto21 = ttest_ind(allsubjectsdata_above30, allsubjectsdata_upto21, equal_var=False)
print('p value between Above30 and Upto21 group = ' + str(p_Above30vsUpto21[2]))
print('t value between Above30 and Upto21 group = ' + str(t_Above30vsUpto21[2]))
print(' ')
t_upto30vsUpto21, p_upto30vsUpto21 = ttest_ind(allsubjectsdata_upto30, allsubjectsdata_upto21, equal_var=False)
print('p value between Upto30 and Upto21 group = ' + str(p_upto30vsUpto21[2]))
print('t value between Upto30 and Upto21 group = ' + str(t_upto30vsUpto21[2]))


#%% Run the statistics for three group using Anova
a1= allsubjectsdata_above30[:,2]
a2= allsubjectsdata_upto30[:,2]
a3= allsubjectsdata_upto21[:,2]

a1_n=a1;
for x in range(2745):
    a1_n = np.append(a1_n, [np.nan])
    
a3_n=a3;
for x in range(2436):
    a3_n = np.append(a3_n, [np.nan])
    
data_latency = {'Above_25':a1_n, 'Upto 25': a2, 'Upto 21': a3_n}
#data_latency = {'Above_25':allsubjectsdata_above30[:,2], 'Upto 25': allsubjectsdata_upto30[:,2], 'Upto 21': allsubjectsdata_upto21[:,2]}
df_latency = pd.DataFrame(data_latency)

import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols('latency ~ C(age)', data=df).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
aov_table
