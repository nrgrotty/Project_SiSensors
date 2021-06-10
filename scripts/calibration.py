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
ax.plot(adcs,n_elect,'o',label='Calibration data')
ax.plot(adcs,estimate_curve,label='Calibration fit')
ax.set(xlabel='ADC charge',ylabel='Number of electrons')
ax.legend()
fig.show()