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
import statsmodels.api as sm
from statsmodels.formula.api import ols

#%% In[131]:


subjectdir_above25 = "C:/Users/bkbme/Desktop/Sneha_program/data_analysis/age/above25/"
subjectdir_upto25 = "C:/Users/bkbme/Desktop/Sneha_program/data_analysis/age/upto25/"
subjectdir_upto21 = "C:/Users/bkbme/Desktop/Sneha_program/data_analysis/age/upto21/"



# In[136]:


allDir_above25 = glob.glob(subjectdir_above25 + os.sep + "*" + os.sep)
allDir_upto25 = glob.glob(subjectdir_upto25 + os.sep + "*" + os.sep)
allDir_upto21 = glob.glob(subjectdir_upto21 + os.sep + "*" + os.sep)

#%% to extract all the subject data above 25
allsubjectsdata_above25 = []
subject_num_above25 = 0
for parentdir in allDir_above25:
    allTrialsSubject_above25 = glob.glob(parentdir + "*.csv")
    for trsubj in allTrialsSubject_above25:
        try:
            data_above25 = np.genfromtxt(trsubj, delimiter=';')
            data_above25 = data_above25[:, :5]
            data_above25 = np.column_stack((data_above25, np.ones((data_above25.shape[0], 1))*subject_num_above25))
            allsubjectsdata_above25.append(np.array(data_above25))
            print(parentdir)
            print(subject_num_above25)
        except:
            print("BAD " + str(trsubj))
    subject_num_above25 += 1

# In[137]:
allsubjectsdata_above25 = np.row_stack(allsubjectsdata_above25)


#%% to extract all the subject data upto 25
allsubjectsdata_upto25 = []
subject_num_upto25 = 0
for parentdir in allDir_upto25:
    allTrialsSubject_upto25 = glob.glob(parentdir + "*.csv")
    for trsubj in allTrialsSubject_upto25:
        try:
            data_upto25 = np.genfromtxt(trsubj, delimiter=';')
            data_upto25 = data_upto25[:, :5]
            data_upto25 = np.column_stack((data_upto25, np.ones((data_upto25.shape[0], 1))*subject_num_upto25))
            allsubjectsdata_upto25.append(np.array(data_upto25))
            print(parentdir)
            print(subject_num_upto25)
        except:
            print("BAD " + str(trsubj))
    subject_num_upto25 += 1
# In[137]:
allsubjectsdata_upto25 = np.row_stack(allsubjectsdata_upto25)


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
allsubjects_Latency_above25 = np.zeros(np.unique(allsubjectsdata_above25[:, -1]).shape, dtype=float, order='C')
for s_num in np.unique(allsubjectsdata_above25[:, -1]):
    try:
        good_index_above25 = allsubjectsdata_above25[:, -1] == s_num
        good_data_above25 = allsubjectsdata_above25[good_index_above25, :]
        good_index_above25 = np.isfinite(good_data_above25[:, 2] - good_data_above25[:, 1])
        allsubjects_Latency_above25[int(s_num)] = np.mean(good_data_above25[:, 2])
        good_data_above25 = good_data_above25[good_index_above25, :]
        hist, bin_edges = np.histogram(good_data_above25[:, 2]/1e6, 15)
        plt.plot(bin_edges[:-1], hist, label=str(good_data_above25[0, -1]))
    except:
        print(s_num)
plt.legend()
plt.show()

#%% to plot latency graph for every subjects upto 25
allsubjects_Latency_upto25 = np.zeros(np.unique(allsubjectsdata_upto25[:, -1]).shape, dtype=float, order='C')
for s_num in np.unique(allsubjectsdata_upto25[:, -1]):
    try:
        good_index_upto25 = allsubjectsdata_upto25[:, -1] == s_num
        good_data_upto25 = allsubjectsdata_upto25[good_index_upto25, :]
        good_index_upto25 = np.isfinite(good_data_upto25[:, 2] - good_data_upto25[:, 1])
        allsubjects_Latency_upto25[int(s_num)] = np.mean(good_data_upto25[:, 2])
        good_data_upto25 = good_data_upto25[good_index_upto25, :]
        hist, bin_edges = np.histogram(good_data_upto25[:, 2]/1e6, 15)
        plt.plot(bin_edges[:-1], hist, label=str(good_data_upto25[0, -1]))
    except:
        print(s_num)
plt.legend()
plt.show()

#%% to plot latency graph for every subjects upto 21
allsubjects_Latency_upto21 = np.zeros(np.unique(allsubjectsdata_upto21[:, -1]).shape, dtype=float, order='C')
for s_num in np.unique(allsubjectsdata_upto21[:, -1]):
    try:
        good_index_upto21 = allsubjectsdata_upto21[:, -1] == s_num
        good_data_upto21 = allsubjectsdata_upto21[good_index_upto21, :]
        good_index_upto21 = np.isfinite(good_data_upto21[:, 2] - good_data_upto21[:, 1])
        allsubjects_Latency_upto21[int(s_num)] = np.mean(good_data_upto21[:, 2])
        good_data_upto21 = good_data_upto21[good_index_upto21, :]
        hist, bin_edges = np.histogram(good_data_upto21[:, 2]/1e6, 15)
        plt.plot(bin_edges[:-1], hist, label=str(good_data_upto21[0, -1]))
    except:
        print(s_num)
plt.legend()
plt.show()

#%% to plot histogram of every group latency
plt.hist(allsubjectsdata_above25[:,2])
plt.show()
plt.hist(allsubjectsdata_upto25[:,2])
plt.show()
plt.hist(allsubjectsdata_upto21[:,2])
plt.show()
#%% Mean,median and standard deviation of latency for aging group and statistics between group
allsubjectsdata_above25_mean = np.mean(allsubjectsdata_above25[:,2])
print('Mean of Latency values of subjects above 25 = ' + str(allsubjectsdata_above25_mean))
allsubjectsdata_above25_median = np.median(allsubjectsdata_above25[:,2])
print('Median of Latency values of subjects above 25 = ' + str(allsubjectsdata_above25_median))
allsubjectsdata_above25_stdev = np.std(allsubjectsdata_above25[:,2])
print('Stdev of Latency values of subjects above 25 = ' + str(allsubjectsdata_above25_stdev))

print('')

allsubjectsdata_upto25_mean = np.mean(allsubjectsdata_upto25[:,2])
print('Mean of Latency values of subjects upto 25 = ' + str(allsubjectsdata_upto25_mean))
allsubjectsdata_upto25_median = np.median(allsubjectsdata_upto25[:,2])
print('Median of Latency values of subjects upto 25 = ' + str(allsubjectsdata_upto25_median))
allsubjectsdata_upto25_stdev = np.std(allsubjectsdata_upto25[:,2])
print('Stdev of Latency values of subjects upto 25 = ' + str(allsubjectsdata_upto25_stdev))

print(' ')

allsubjectsdata_upto21_mean = np.mean(allsubjectsdata_upto21[:,2])
print('Mean of Latency values of subjects upto 21 = ' + str(allsubjectsdata_upto21_mean))
allsubjectsdata_upto21_median = np.median(allsubjectsdata_upto21[:,2])
print('Median of Latency values of subjects upto 21 = ' + str(allsubjectsdata_upto21_median))
allsubjectsdata_upto21_stdev = np.std(allsubjectsdata_upto21[:,2])
print('Stdev of Latency values of subjects upto 21 = ' + str(allsubjectsdata_upto21_stdev))

#%% calculate p,t values of every aging groups using simple t-test
t_Above25vsUpto25, p_Above25vsUpto25 = ttest_ind(allsubjectsdata_above25, allsubjectsdata_upto25, equal_var=False)
print('p value between Above25 and Upto25 group = ' + str(p_Above25vsUpto25[2]))
print('t value between Above25 and Upto25 group = ' + str(t_Above25vsUpto25[2]))
print(' ')
t_Above25vsUpto21, p_Above25vsUpto21 = ttest_ind(allsubjectsdata_above25, allsubjectsdata_upto21, equal_var=False)
print('p value between Above25 and Upto21 group = ' + str(p_Above25vsUpto21[2]))
print('t value between Above25 and Upto21 group = ' + str(t_Above25vsUpto21[2]))
print(' ')
t_upto25vsUpto21, p_upto25vsUpto21 = ttest_ind(allsubjectsdata_upto25, allsubjectsdata_upto21, equal_var=False)
print('p value between Upto25 and Upto21 group = ' + str(p_upto25vsUpto21[2]))
print('t value between Upto25 and Upto21 group = ' + str(t_upto25vsUpto21[2]))

#%% Run the statistics for three group using Anova
# creat Data frame for the thre group latency
a1= allsubjectsdata_above25[:,2]
a2= allsubjectsdata_upto25[:,2]
a3= allsubjectsdata_upto21[:,2]

a1_n=a1;
for x in range(2745):
    a1_n = np.append(a1_n, [np.nan])
    
a3_n=a3;
for x in range(2436):
    a3_n = np.append(a3_n, [np.nan])
    
data_latency = {'Above_25':a1_n, 'Upto 25': a2, 'Upto 21': a3_n}

#data_latency = {'Above_25':allsubjectsdata_above25[:,2], 'Upto 25': allsubjectsdata_upto25[:,2], 'Upto 21': allsubjectsdata_upto21[:,2]}
df_latency = pd.DataFrame(data_latency)

# build the model and use the Anova test for three groups
model = ols('latency ~ C(age)', data=df).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
aov_table
