#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 17:00:39 2021

@author: neus
"""
from ExternalFunctions import *
from alibaba_functions import get_clusters,get_calibration_function,fit_chargeDistribution,plot_clusters
import pandas as pd

fig_path = '../figures/'
files_path = '../data/'
results_path = '../results/'

signal_dataFiles_names = ['signaldata']
SNR_cuts = [5]

for i in range(len(signal_dataFiles_names)):   
    clusters = get_clusters(files_path+signal_dataFiles_names[i],SNR_cuts[i])
    
    charge = pd.DataFrame(clusters['cluster_charge'],columns=['Charge'])
    charge.to_csv(results_path+'charge_'+signal_dataFiles_names[i],index=False)
    
    #Make the plots to check that the SNR_cut is fine
    plot_clusters(clusters['cluster_size'],clusters['n_clusters'],
                  fig_path=fig_path+signal_dataFiles_names[i],SNR_cut=SNR_cuts[i])
    
