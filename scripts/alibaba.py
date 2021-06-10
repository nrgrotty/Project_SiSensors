# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from ExternalFunctions import *
import more_itertools as mit
from iminuit import Minuit  
from scipy import stats

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

fig_path = '../figures/'
files_path = '../data/'


# In[EXERCISE 4]

filename = files_path+'signaldata.h5'
F = h5py.File(filename, "r") 

filename2 = files_path+'Data100'
F2 = h5py.File(filename, "r") 

raw_data_noSignal = F2['events/signal']

pedestals_estimate = F['header/pedestal'][0]
noise_estimate = F['header/noise'][0]
raw_data = F['events/signal']

common_mode = np.sum((raw_data-pedestals_estimate),axis=1)/128 
signal = np.array(raw_data) - np.array(pedestals_estimate) - np.array([common_mode]).T

#signal = get_signal(raw_data,raw_data_noSignal)
#noise= get_commonNoise(raw_data_noSignal)

SNR = signal/noise_estimate
SNR_cut = 4

# print(SNR.shape)
# print(SNR[:6,:3])

N_events,N_channels = raw_data.shape 

clusters_charge = []
clusters_size = []

n_clusters_per_event = []
for k in range(N_events):
    
    SNR_event = SNR[k,:]
    SNR_greater = [i for i, value in enumerate(SNR_event) if value>SNR_cut] #list of indices where the SNR is greater than the SNR_cut
    
    n_clusters = 0
    
    for cluster in mit.consecutive_groups(SNR_greater):
        
        indexes_cluster = list(cluster)
        
        charge = np.sum( signal[k,indexes_cluster] )
        cluster_size = len(indexes_cluster)
        
        clusters_charge.append(charge)
        clusters_size.append(cluster_size)
        
        n_clusters += 1
        
        # if (k<10):
        #     print(f'Event = {k} \t Cl.channels = {indexes_cluster} \t Cl.charges = {charge} \t n_clust.={n_clusters}')

    
    n_clusters_per_event.append(n_clusters)

clusters_charge = np.array(clusters_charge)        
        
fig,ax = plt.subplots(1,2,figsize=(10,4))

max_nclustevents = np.max(n_clusters_per_event)
create_1d_hist(ax[0],n_clusters_per_event,bins=max_nclustevents+1,x_range=(-0.5,max_nclustevents+0.5),title='N. clusters / event')
ax[0].set(ylabel='Counts', xlabel='N clusters')

max_sizes = np.max(clusters_size)
create_1d_hist(ax[1],clusters_size,bins=max_sizes+1,x_range=(-0.5,max_sizes+0.5),title='Cluster sizes')
ax[1].set(ylabel='Counts',xlabel='N strips per cluster')

fig.suptitle(f'SNR_cut = {SNR_cut:1.2f}',fontsize=20)
fig.tight_layout()
fig.savefig(fig_path+'./cluster_histograms.png')


fig_charge, ax_charge = plt.subplots()

max_charge,min_charge = np.max(clusters_charge),np.min(clusters_charge)
Nbins_charge = 130

y, edges = create_1d_hist(ax_charge,clusters_charge,bins=Nbins_charge,x_range=(min_charge,400),title='Clusters charge')
ax_charge.set(yscale='linear')
ax_charge.set(xlabel='ADC charge',ylabel='Counts')



# In[]

signal = signal.flatten()
sig_min,sig_max = np.min(signal),np.max(signal)

fig,ax = plt.subplots()
create_1d_hist(ax,signal,bins=100,x_range=(sig_min,sig_max),title='Raw signal')
ax.set(yscale='log')

fig.tight_layout()
fig.savefig(fig_path+'./signa_hist.png')


ch_min,ch_max = np.min(clusters_charge),np.max(clusters_charge)

#plt.show(block=True)

# In[Fitting]

lower_th,upper_th = 80,145

x = (edges[:-1] + edges[1:])/2
sy = np.sqrt(y)
bin_width = (max_charge-min_charge)/Nbins_charge

# x = x[5:45]
# y = y[5:45]
# sy = sy[5:45]

#Fiting with chi2
mask = (x>lower_th)&(x<upper_th)
chi2_object = Chi2Regression(moyal_distr_extended,x[mask],y[mask],sy[mask])
minuit_chi2 = Minuit(chi2_object,N=len(clusters_charge),binwidth=bin_width,m=np.mean(clusters_charge),s=np.std(clusters_charge))#),pedantic=False,print_level=0,fix_binwidth=True)
minuit_chi2.migrad();

chi2_value = minuit_chi2.fval
Ndof = len(y[mask])-minuit_chi2.nfit#(len(minuit_chi2.values)-1)
Pchi2 = stats.chi2.sf(chi2_value,Ndof)

moyal_distr_extended_vec = np.vectorize(moyal_distr_extended)
y_fit_chi2 = moyal_distr_extended_vec(x[mask],minuit_chi2.values['N'],minuit_chi2.values['binwidth'],minuit_chi2.values['m'],minuit_chi2.values['s'])

format_mu="{:1.2f}+/-{:1.2f}"
format_N="{:1.0f}+/-{:1.0f}"
format_sigma="{:1.2f}+/-{:1.2f}"
format_p="{:.4f}%"
format_binwidth="{:1.3f}+/-{:1.3f}"
    
d_chi2 = {'N': format_N.format(minuit_chi2.values['N'], minuit_chi2.errors['N']),
    'binwidth': format_binwidth.format(minuit_chi2.values['binwidth'],minuit_chi2.errors['binwidth']),
     r'$\mu$': format_mu.format(minuit_chi2.values['m'],minuit_chi2.errors['m']),
     r'$\sigma$': format_sigma.format(minuit_chi2.values['s'],minuit_chi2.errors['s']),
     r'$\chi^2$/Ndof': "{:.2f}/{:d}".format(chi2_value,Ndof),
     r'P($\chi^2$)': format_p.format(Pchi2*100)}    
#Fitting with unbinned likelihood

lower_th,upper_th=50,180
mask_ullh = (clusters_charge>lower_th)&(clusters_charge<upper_th)
data_to_fit = clusters_charge[mask_ullh]

ullh_object = UnbinnedLH(moyal_distr_extended,data_to_fit,weights=None)
minuit_ullh = Minuit(ullh_object,N=len(data_to_fit),binwidth=bin_width,m=np.mean(data_to_fit),s=np.std(data_to_fit))#,fix_binwidth=True,fix_N=True,pedantic=False,print_level=0)
minuit_ullh.migrad()

y_fit_ullh = moyal_distr_extended_vec(x[mask],minuit_ullh.values['N'],minuit_ullh.values['binwidth'],minuit_ullh.values['m'],minuit_ullh.values['s'])/1e18#(minuit_ullh.values['binwidth']*minuit_ullh.values['N'])

d_ullh = {'N': format_N.format(minuit_ullh.values['N'], minuit_ullh.errors['N']),
     r'$\mu$': format_mu.format(minuit_ullh.values['m'],minuit_ullh.errors['m']),
     r'$\sigma$': format_sigma.format(minuit_ullh.values['s'],minuit_ullh.errors['s']),}


add_text_to_ax(0.42,0.55,nice_string_output(d_ullh,0),ax_charge,fontsize=12,color='green')
add_text_to_ax(0.42,0.95,nice_string_output(d_chi2,0),ax_charge,fontsize=12,color='blue')
    
ax_charge.plot(x[mask],y_fit_chi2,'-o',label='chi2 fit',color='blue')
ax_charge.plot(x[mask],y_fit_ullh,label=r'ULLH fit',color='green')

fig_charge.tight_layout()
fig_charge.savefig(fig_path+'./cluster_charge_hist.png')

print(minuit_chi2.values)
print(minuit_ullh.values)
print(chi2_value,Pchi2)


plt.show(block=True)