#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 17:00:39 2021

@author: neus
"""
from ExternalFunctions import *
from alibaba_functions import get_clusters,get_calibration_function,fit_chargeDistribution,plot_clusters
import pandas as pd
import sys

fig_path = '../figures/'
files_path = '../CCE_DATA/'
results_path = '../results/'


signal_dataFiles_names = ['data_010V','data_020V','data_030V','data_040V','data_050V','data_060V',
                          'data_070V','data_080V','data_090V','data_100V','data_110V']
SNR_cuts = [4,4,4,3,4,4,
            4,4,4,4,4]

for i in range(len(signal_dataFiles_names)):   
    clusters = get_clusters(files_path+signal_dataFiles_names[i],SNR_cuts[i])
    
    charge = pd.DataFrame(clusters['cluster_charge'],columns=['Charge[ADC]'])
    charge.to_csv(results_path+'charge_'+signal_dataFiles_names[i],index=False)
    
    #Make the plots to check that the SNR_cut is fine
    plot_clusters(clusters['cluster_size'],clusters['n_clusters'],
                  fig_path=fig_path+signal_dataFiles_names[i]+str(SNR_cuts[i]),SNR_cut=SNR_cuts[i])
    
    # #Make spectrum 
    # fig,ax = plt.subplots()
    # ax.hist(clusters['cluster_charge'],bins=100)
    # ax.set(yscale='log')
    # fig.suptitle(f'SNR_cut = {SNR_cuts[i]:1.2f}',fontsize=20)
    # fig.savefig(fig_path+'spectrum_'+signal_dataFiles_names[i]+str(SNR_cuts[i]))