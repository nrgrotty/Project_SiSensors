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

fig,ax=plt.subplots()
ax.plot(-voltage[1:],c2_inv[1:],'-o')
ax.set(xlabel='Reverse bias [V]',ylabel=r'$1/C^{2}$[pF$^{-2}$]')


# In[fit CV]

def fitted_line(x,a,b):
    return a*x+b

v_fit1 = -voltage[1:6]
c2_fit1 = c2_inv[1:6]

coefs1,cov1 = np.polyfit(v_fit1,c2_fit1,1,cov=True,full=False)
x_fit1 = range(70)
ax.plot(x_fit1,fitted_line(x_fit1,coefs1[0],coefs1[1]),label='Fitted line',color='orange')

v_fit2 = -voltage[10:]
c2_fit2 = c2_inv[10:]

coefs2,cov2 = np.polyfit(v_fit2,c2_fit2,1,cov=True,full=False)
x_fit2 = range(60,np.max(v_fit2))
ax.plot(x_fit2,fitted_line(x_fit2,coefs2[0],coefs2[1]),label='Fitted line',color='green')

fig.tight_layout()
fig.savefig('../figures/CV_lab.png')
