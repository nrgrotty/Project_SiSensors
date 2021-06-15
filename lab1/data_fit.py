import pandas as pd
import numpy as np
import logging
from matplotlib import pyplot as plt



# In[Leakage current]
data = pd.read_csv('measurement4.txt')

length = len(data['current[A]'])

var = np.split(data['current[A]'],length/10)
volt = np.split(data['voltage[V]'],length/10)
volt_mean = np.mean(volt,axis=1)
var_mean = np.mean(var, axis = 1)
print(volt_mean)
print(var_mean)


fig,ax=plt.subplots()
ax.plot(-volt_mean[1:],-var_mean[1:],'-o')
ax.set(xlabel='Reverse bias [V]',ylabel='Current[A]')
fig.tight_layout()
fig.savefig('../figures/IV_lab.png')


# In[Depletion voltage]

data = pd.read_csv('../data/C_measurement_10_06.txt')

c2_inv = 1/data['Capacitance[pF]']**2
voltage = data['Voltage[V]']

fig,ax=plt.subplots(figsize=(6,3))
ax.plot(-voltage[1:],c2_inv[1:],'-o')
ax.set(xlabel='Reverse bias [V]',ylabel=r'$1/C^{2}$[pF$^{-2}$]')                                          


# In[fit CV]

def fitted_line(x,a,b):
    return a*x+b

v_fit1 = -voltage[1:6]
c2_fit1 = c2_inv[1:6]

coefs1,cov1 = np.polyfit(v_fit1,c2_fit1,1,cov=True,full=False)
delta_a1,delta_b1 = np.sqrt(cov1[0,0]),np.sqrt(cov1[1,1])

x_fit1 = range(70)
ax.plot(x_fit1,fitted_line(x_fit1,coefs1[0],coefs1[1]),label='Fitted line',color='red')
ax.text(0.17, 0.2, r'1/$C^2$'+f' = ({coefs1[0]:2.3e} +\- {delta_a1:2.0e})  V + ({coefs1[1]:2.1e} +\- {delta_b1:2.0e}) ', transform=ax.transAxes,color='red')

v_fit2 = -voltage[9:]
c2_fit2 = c2_inv[9:]

coefs2,cov2 = np.polyfit(v_fit2,c2_fit2,1,cov=True,full=False)
delta_a2,delta_b2 = np.sqrt(cov2[0,0]),np.sqrt(cov2[1,1])

x_fit2 = range(60,np.max(v_fit2))
ax.plot(x_fit2,fitted_line(x_fit2,coefs2[0],coefs2[1]),label='Fitted line',color='green')
ax.text(0.17, 0.3, r'1/$C^2$'+f' = ({coefs2[0]:2.0e} +\- {delta_a2:2.0e}) V + ({coefs2[1]:2.3e} +\- {delta_b2:2.0e}) ', transform=ax.transAxes,color='green')

v_dep = (coefs2[1]-coefs1[1])/(coefs1[0]-coefs2[0])
delta_vdep = np.sqrt( v_dep**2*(delta_a1**2+delta_a2**2)+(delta_b1**2+delta_b2**2) ) / (coefs1[0]-coefs2[0])

ax.text(0.17, 0.1, r'$V_{dep}$'+f' = ({v_dep:3.1f} +\- {delta_vdep:3.1f}) V', transform=ax.transAxes,color='black')
ax.plot(v_dep,coefs1[0]*v_dep+coefs1[1],'o',color='black')

fig.tight_layout()
fig.savefig('../figures/CV_lab.png')
