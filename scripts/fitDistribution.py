#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 17:13:38 2021

@author: neus
"""
from ExternalFunctions import *
from alibaba_functions import get_clusters,get_calibration_function,fit_chargeDistribution,plot_clusters
import pandas as pd
import matplotlib.pyplot as plt

fig_path = '../figures/'
files_path = '../data/'
results_path = '../results/'

# signal_dataFiles_names = ['data_010V','data_020V','data_030V','data_040V','data_050V','data_060V',
#                           'data_070V','data_080V','data_090V','data_100V','data_110V']

fit_parameters = {#filename : [volt,low_th,high_th,Nbins]
                    '010V':[10,17,40,150],
                   '020V':[20,25,50,120],
                   '030V':[30,35,70,120],
                   '040V':[40,45,80,140],
                   '050V':[50,55,80,190],
                   '060V':[60,63,100,140],
                    '070V':[70,68,110,120],
                    '080V':[80,70,120,160], 
                   '090V':[90,75,110,180],
                   '100V':[100,72,120,130],
                   '110V':[110,75,120,180],
                   }
voltages = []
adc_means = []
adc_errors = []

for file_name in fit_parameters.keys():
    
    voltages.append(fit_parameters[file_name][0])
    low_th,high_th,Nbins = fit_parameters[file_name][1],fit_parameters[file_name][2],fit_parameters[file_name][3]
    
    charge = pd.read_csv(results_path+'charge_'+file_name)['Charge[keV]']
    fitting_results = fit_chargeDistribution(charge,low_th,high_th,fig_path=fig_path+file_name,
                                             Nbins_charge=Nbins,title=file_name,units='keV')
    
    adc_means.append(fitting_results['m'])
    adc_errors.append(fitting_results['error_m'])
 
# In[Depletion voltage before calibration]

fig,ax=plt.subplots()
ax.plot(voltages,(np.array(adc_means)/adc_means[-1])**2,'-o')
ax.set(xlabel='Voltage[V]',ylabel='Charge[keV]')
fig.show()

