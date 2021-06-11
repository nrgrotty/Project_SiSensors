import pandas as pd
import numpy as np
import logging
from matplotlib import pyplot as plt



data = pd.read_csv('C_measurement_10_06.txt')
#data = np.genfromtxt('C_measurement_10_06.txt', )
#length = len(data['Voltage[V]'])


print(data)

volt = (-1)*data['Voltage[V]'][1:]
c = data[' Capacitance[pF]'][1:]

c2 = np.square(c)

c_final = 1/c2

#var = np.split(data['current[A]'],length/10)
#volt = np.split(data['voltage[V]'],length/10)
#olt_mean = np.mean(volt,axis=1)
#var_mean = np.mean(var, axis = 1)
#print(volt_mean)
#print(var_mean)


plt.plot(volt,c_final,'-o')
plt.xlabel('-V[V]reverse bias')
plt.ylabel('-capacitance[pF]')
plt.show()

