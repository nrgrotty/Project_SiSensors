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
from ExternalFunctions import create_1d_hist,plot_fit_gaussian
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

fig_path = '../figures/'
files_path = '../data/'

voltage=np.array([10,20,30,40,50,60,70,80,90,100])
current=np.array([0.,0.1,0.22,0.54,0.61,0.64,0.67,0.68,0.69,0.7])

fig,ax = plt.subplots()
ax.plot(voltage,current,'-o')
ax.set(xlabel='Voltage [V]',ylabel=r'Current [$\mu$A]')
fig.tight_layout()
fig.savefig(fig_path+'leakage_currnet_alibava.png')

# In[EXERCISE 2] 

filenames = ['Data1','Data2','Data3','Data4','Data5','Data7','Data10',
              'Data14','Data19','Data27','Data37','Data52','Data72','Data100']
voltages = [1,2,3,4,5,7,10,14,19,27,37,52,72,100]


fig_distr,ax_distr = plt.subplots()

means_noise,std_noise = [],[]
means_noise_central,std_noise_central = [],[]
for filename in filenames:

    F = h5py.File(files_path+filename, "r") 
    pedestals_estimate = np.array(F.get('header/pedestal')[0])
    raw_data = np.array(F.get('events/signal'))

    common_noise = get_commonNoise(raw_data)
    ax_distr.hist(common_noise,100,alpha=0.5,label=filename[4:]+'V')
    
    means_noise.append(np.mean(common_noise))
    std_noise.append(np.std(common_noise))
    
    means_noise_central.append(np.mean(common_noise[40:-40]))
    std_noise_central.append(np.std(common_noise[40:-40]))

ax_distr.set(xlabel='Noise [ADC]',ylabel='Counts')
ax_distr.legend()
fig_distr.tight_layout()
fig_distr.savefig(fig_path+'common_noise_distr.png')

fig,ax = plt.subplots()
ax.errorbar(voltages,means_noise,yerr=np.array(std_noise),fmt='-o',label='All channels')
ax.set(xlabel='Voltage [V]', ylabel='Mean noise [ADC]')
fig.savefig(fig_path+'mean_noise_voltage_alibava.png')


# In[get more plots]

F = h5py.File(files_path+'Data52', "r") 
raw_data = np.array(F.get('events/signal'))

common_mode = get_commonMode(raw_data)

N_bins = 107
fig,ax = plt.subplots(2,1,figsize=(8,5))

ax[0].plot(common_mode,label='All channels')
ax[0].set(xlabel='Event', ylabel='Common mode [ADC]')

ax[1].hist(common_mode,bins=N_bins,alpha=0.6)
ax[1].set(xlabel='Common mode [ADC]',ylabel='Counts')

plot_fit_gaussian(common_mode,N_bins,ax=ax[1],fit_ullh=False,format_p='{:1.2e}')

fig.tight_layout()
fig.savefig(fig_path+'commonMode_alibava_52.png')


# In[]

F = h5py.File(files_path+'Data2', "r") 
raw_data = np.array(F.get('events/signal'))

signal = get_signal(raw_data,raw_data)[:,45]

N_bins = 107
fig,ax = plt.subplots(2,1,figsize=(8,5))

ax[0].plot(signal,label='All channels')
ax[0].set(xlabel='Event', ylabel='Signal [ADC] (Channel 44)')

ax[1].hist(signal,bins=N_bins,alpha=0.6)
ax[1].set(xlabel='Signal [ADC] (Channel 44)',ylabel='Counts')

plot_fit_gaussian(signal,N_bins,ax=ax[1],fit_ullh=False)#,format_p='{:1.2e}')

fig.tight_layout()
fig.savefig(fig_path+'signal_alibava_2.png')

