#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 17:13:38 2021

@author: neus
"""
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
charge = pd.read_csv(results_path+file_name)['Charge']
fitting_results = fit_chargeDistribution(charge,75,145,fig_path=fig_path+file_name,Nbins_charge=130)


adc_means.append(fitting_results['m'])
adc_errors.append(fitting_results['error_m'])

