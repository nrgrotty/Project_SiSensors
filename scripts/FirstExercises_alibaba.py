#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 10:03:06 2021

@author: neus
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from ExternalFunctions import create_1d_hist
import more_itertools as mit

def get_pedestals(raw_data):    
    N_events, N_channels = raw_data.shape
    return np.sum(raw_data,axis=0)/N_events

def get_commonMode(raw_data):
    N_events, N_channels = raw_data.shape
    p_i = get_pedestals(raw_data)   
    return np.sum((raw_data-p_i),axis=1)/N_channels  

def get_signal(raw_data,raw_data_noSignal):   
    p_i = get_pedestals(raw_data_noSignal)
    d_k = get_commonMode(raw_data_noSignal)
    return raw_data - np.array([p_i])-np.array([d_k]).T

def get_commonNoise(raw_data):
    return np.std(get_signal(raw_data,raw_data),axis=0)

# In[EXERCISE 1]

voltage=np.array([10,20,30,40,50,60,70,80,90,100])
current=np.array([0.,0.1,0.22,0.54,0.61,0.64,0.67,0.68,0.69,0.7])

#fig,ax = plt.subplots()
#ax.plot(voltage,current,'-o')
#ax.set(xlabel='Voltage[V]',ylabel=r'Current[$\mu$A]')
#fig.show()

# In[EXERCISE 2] 

filenames = ['Data1','Data2','Data3','Data4','Data5','Data7','Data10',
              'Data14','Data19','Data27','Data37','Data52','Data72','Data100']
voltages = [1,2,3,4,5,7,10,14,19,27,37,52,72,100]

means_noise,std_noise = [],[]
means_noise_central,std_noise_central = [],[]
for filename in filenames:

    F = h5py.File(filename, "r") 
    pedestals_estimate = np.array(F.get('header/pedestal')[0])
    raw_data = np.array(F.get('events/signal')[0])

    common_noise = get_commonNoise(raw_data)

    means_noise.append(np.mean(common_noise))
    std_noise.append(np.std(common_noise))
    
    means_noise_central.append(np.mean(common_noise[40:-40]))
    std_noise_central.append(np.std(common_noise[40:-40]))

fig,ax = plt.subplots()
ax.errorbar(voltages,means_noise,yerr=np.array(std_noise),fmt='-o',label='All channels')
ax.errorbar(voltages,means_noise,yerr=np.array(std_noise),fmt='-o',label='From channel 40 to 87')
ax.legend()
ax.set(xlabel='Voltage [V]', ylabel='Mean noise [ADC]')
fig.savefig('./noise_voltage.png')