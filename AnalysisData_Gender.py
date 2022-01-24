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


# In[131]:


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
#%% Mean and standard, median daviation of latency for male and female group and statistics between group


#%%
allsubjectsdata_latency = np.concatenate((allsubjects_Latency_above25, allsubjects_Latency_upto25, allsubjects_Latency_upto21), axis=0)
allsubjectsdata_latency1 = np.concatenate((allsubjects_Latency_above25, allsubjects_Latency_upto25), axis=0)
#%%
allsubjects_age_above25 = np.zeros(np.unique(allsubjectsdata_above25[:, -1]).shape, dtype=float, order='C')
i=0
for parentdir in allDir_above25:
    allsubjects_age_above25[i]=int(parentdir[-3:-1])
    i=i+1
    
allsubjects_age_upto25 = np.zeros(np.unique(allsubjectsdata_upto25[:, -1]).shape, dtype=float, order='C')
i=0
for parentdir in allDir_upto25:
    allsubjects_age_upto25[i]=int(parentdir[-3:-1])
    i=i+1
    
allsubjects_age_upto21 = np.zeros(np.unique(allsubjectsdata_upto21[:, -1]).shape, dtype=float, order='C')
i=0
for parentdir in allDir_upto21:
    allsubjects_age_upto21[i]=int(parentdir[-3:-1])
    i=i+1
    
allsubjectsdata_age = np.concatenate((allsubjects_age_above25, allsubjects_age_upto25, allsubjects_age_upto21), axis=0)
allsubjectsdata_age1 = np.concatenate((allsubjects_age_above25, allsubjects_age_upto25), axis=0)
#%%
plt.scatter(allsubjectsdata_age1, allsubjectsdata_latency1)
plt.show()