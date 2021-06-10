#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 16:26:16 2021

@author: neus
"""
from ExternalFunctions import *
import numpy as np
import matplotlib.pyplot as plt
import h5py
import more_itertools as mit
import pandas as pd
from iminuit import Minuit  
from scipy import stats

def get_clusters(file_path,SNR_cut):
    
    F = h5py.File(file_path, "r") 

    pedestals_estimate = F['header/pedestal'][0]
    noise_estimate = F['header/noise'][0]
    raw_data = F['events/signal']

    common_mode = np.sum((raw_data-pedestals_estimate),axis=1)/128 
    signal = np.array(raw_data) - np.array(pedestals_estimate) - np.array([common_mode]).T
    
    SNR = signal/noise_estimate
    
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
    
    results = {'cluster_charge':clusters_charge,
               'cluster_size':clusters_size,
               'n_clusters':n_clusters_per_event}
    
    return results


def get_calibration_function(file_path,degree=4):
    
    data = pd.read_csv(file_path,skiprows=1,delimiter='\t')

    n_elect = data['X']
    adcs = data['Function_0']
    
    coefs,cov = np.polyfit(adcs,n_elect,degree,cov=True,full=False)

    def calibration_func(x):
    
        poly_deg = coefs.size-1    
        x_powers = np.array([x**(poly_deg-n) for n in range(poly_deg+1)]).T  
        
        return np.dot(x_powers,coefs)
    
    return calibration_func

def plot_clusters(cluster_size,n_clusters,fig_path=None,SNR_cut=-1.):
    fig,ax = plt.subplots(1,2,figsize=(10,4))

    max_nclustevents = np.max(n_clusters)
    create_1d_hist(ax[0],n_clusters,bins=max_nclustevents+1,x_range=(-0.5,max_nclustevents+0.5),title='N. clusters / event')
    ax[0].set(ylabel='Counts', xlabel='N clusters')
    
    max_sizes = np.max(cluster_size)
    create_1d_hist(ax[1],cluster_size,bins=max_sizes+1,x_range=(-0.5,max_sizes+0.5),title='Cluster sizes')
    ax[1].set(ylabel='Counts',xlabel='N strips per cluster')
    
    fig.suptitle(f'SNR_cut = {SNR_cut:1.2f}',fontsize=20)
    fig.tight_layout()
    
    if (fig_path is not None): fig.savefig(fig_path)
    
    return fig,ax

def fit_chargeDistribution(clusters_charge,lower_th,upper_th,max_charge=400,fig_path=None,Nbins_charge=130,units='ADC'):
    
    #Define range to plot
    #max_charge,min_charge = np.max(clusters_charge),np.min(clusters_charge)
    min_charge = np.min(clusters_charge)
    
    #Ploting histogram
    fig_charge, ax_charge = plt.subplots()
    y, edges = create_1d_hist(ax_charge,clusters_charge,bins=Nbins_charge,x_range=(min_charge,max_charge),title='Clusters charge',alpha=0.6)
    ax_charge.set(xlabel=f'Charge [{units}]',ylabel='Counts')#,ylim=(0,3000))
    
    #Getting binned data
    x = (edges[:-1] + edges[1:])/2
    sy = np.sqrt(y)
    bin_width = (max_charge-min_charge)/Nbins_charge


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
    
    add_text_to_ax(0.4,0.95,nice_string_output(d_chi2,0),ax_charge,fontsize=12,color='blue')
        
    ax_charge.plot(x[mask],y_fit_chi2,'-o',label='chi2 fit',color='blue')
    
    fig_charge.tight_layout()
    if (fig_path is not None): fig_charge.savefig(fig_path)
        
    
    results = {'m':minuit_chi2.values['m'] ,
               'error_m':minuit_chi2.errors['m'],
               's':minuit_chi2.values['s'],
               'error_s':minuit_chi2.errors['s'],
               'figure':fig_charge,
               'axis': ax_charge}
    return results
        
    
    

