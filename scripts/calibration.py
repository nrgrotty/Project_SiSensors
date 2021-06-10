#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 12:41:05 2021

@author: neus
"""
import numpy as np
from ExternalFunctions import * 
import pandas as pd 
import matplotlib.pyplot as plt

files_path = '../data/'
fig_path = '../figures/'

data = pd.read_csv(files_path+'Calibration_charge_allRange_hist',skiprows=1,delimiter='\t')

n_elect = data['X']
adcs = data['Function_0']

coefs,cov = np.polyfit(adcs,n_elect,4,cov=True,full=False)

def calibration_func(x,coefs):
    
    poly_deg = coefs.size-1    
    x_powers = np.array([x**(poly_deg-n) for n in range(poly_deg+1)]).T  
    
    return np.dot(x_powers,coefs)

estimate_curve = calibration_func(adcs,coefs)

# estimate_curve=[]
# for x in adcs:
#     estimate_curve.append(calibration_func(x,coefs))

fig,ax=plt.subplots()
ax.plot(adcs,n_elect,'o',label='Data')
ax.plot(adcs,estimate_curve,label='Fit')
ax.set(xlabel='Charge [ADC]',ylabel='Charge [Num.electrons]')
ax.legend()
fig.savefig(fig_path+'calibration_110.png')


# In[Nice plot after calibration]

from ExternalFunctions import *
from alibaba_functions import get_clusters,get_calibration_function,fit_chargeDistribution,plot_clusters
import pandas as pd

fig_path = '../figures/'
files_path = '../data/'
results_path = '../results/'

voltages = [110]
adc_means = []
adc_errors = []


#CODE THAT NEEDS TO BE ADAPTED
file_name = 'signaldata'
chargeADC = pd.read_csv(results_path+'charge_'+file_name)['Charge']
fitting_results = fit_chargeDistribution(chargeADC,80,145,fig_path=fig_path+'_chargeADC_'+file_name,Nbins_charge=130,max_charge=300)


calibration_func = get_calibration_function(files_path+'Calibration_charge_allRange_hist',degree=4)
charge = calibration_func(chargeADC)/3.62

fitting_results = fit_chargeDistribution(charge,5500,10000,fig_path=fig_path+'_charge_'+file_name,Nbins_charge=120,max_charge=30000,units='eV')

adc_means.append(fitting_results['m'])
adc_errors.append(fitting_results['error_m'])
