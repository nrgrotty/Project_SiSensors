import logging
import argparse
import serial
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
###----------------------------------------------------------------------------------------------------------------### 
### INSTRUCTIONS
###----------------------------------------------------------------------------------------------------------------### 
### This script will...

### * set up a connection with the smu
### * configure the smu
### * ramp the voltage up to 10 V @ a rate of 1 V per second
### * wait 10 s for the system to settle
### * take 10 current measurements @ a rate of 1 per second
### * print the measurements to the screen
### * ramp back down to 0 V

### You'll need to update/ improve the script so that it can
### scan through a series of voltage points and take current
### measurements at each point. The START, STOP voltages and
### the voltage STEP should be configurable from the command
### line. You should also write out the measurements to text
### file for later analysis.

### You MUST make use of the ramp function as it's important
### that the voltage is increased/ decreased slowly and must
### always be returned back to 0 V

### Likewise you MUST always make use of the try/except key-
### boardInterrupt as it provides a way to abort the program
### safely (if necessary).
###----------------------------------------------------------------------------------------------------------------###
### logging
logging.basicConfig(format='[%(asctime)s] %(levelname)-7s %(message)s',datefmt='%d/%m/%Y %H:%M:%S',level=logging.INFO)
###----------------------------------------------------------------------------------------------------------------### 
### function to ramp voltage up/down
def ramp(smu, v_end, speed):
    smu.write(':SOUR:VOLT?\r') # check the current voltage
    v = int(float(smu.readline()))
    if v < v_end: # ramp up
        while v < v_end:
            v += 1
            time.sleep(speed)
            logging.info("ramping voltage to %s V" % v)
            smu.write(':SOUR:VOLT %s\r' % v)
    if v > v_end: # ramp down
        while v > v_end:
            v -= 1
            time.sleep(speed)
            logging.info("ramping voltage to %s V" % v)
            smu.write(':SOUR:VOLT %s\r' % v)
###----------------------------------------------------------------------------------------------------------------### 
### parse the given script arguments
parser = argparse.ArgumentParser()

# optional arguments
parser.add_argument("--number_meas", type=  int, default=10)
parser.add_argument("--time_delay",  type=float, default=10)
parser.add_argument("--meas_speed",  type=float, default= 1)
parser.add_argument("--ramp_speed",  type=float, default= 1)
parser.add_argument("--sweep_back",  action="store_true")
parser.add_argument("--rear_terms",  action="store_true")
parser.add_argument("--NPLC",        default="1")

args = parser.parse_args()

###----------------------------------------------------------------------------------------------------------------### 
### set up a connection with the smu
smu = serial.Serial('/dev/ttyUSB0')
smu.timeout = 10
smu.write('*IDN?\r')
idn = smu.readline()
logging.info(idn)
#print smu.read
###----------------------------------------------------------------------------------------------------------------### 
### configure the smu
smu.write('*RST\r')
smu.write(':SOUR:FUNC VOLT\r')
smu.write(':SOUR:VOLT:MODE FIXED\r')
smu.write(':SOUR:VOLT:PROT 1001\r')
smu.write(':SOUR:VOLT:RANG MAX\r')
smu.write(':SENS:FUNC "CURR"\r')
smu.write(':SENS:CURR:PROT 1E-8\r')
smu.write(':SENS:CURR:RANG MIN\r')
smu.write(':SENS:CURR:NPLC %s\r' % args.NPLC)
if args.rear_terms:
    smu.write(':ROUT:TERM REAR\r')
smu.write(':FORM:ELEM VOLT, CURR\r')
###----------------------------------------------------------------------------------------------------------------### 
### set voltage and take current measurements
#try:
#    smu.write(':OUTP ON\r')
#    logging.info("preparing for measurement(s) at 10 V")
#    ramp(smu, 10, args.ramp_speed) ## ramp up to 10 V
#    time.sleep(args.time_delay)
#    for n in range(args.number_meas):
#        logging.info("taking measurement")
#        smu.write(':READ?\r')
#        data = smu.readline()
#        logging.info(data)
#        time.sleep(args.meas_speed)
#    ramp(smu, 0, args.ramp_speed) ## ramp down to 0 V
#    smu.write(':OUTP OFF\r')
#
#except KeyboardInterrupt:
#    logging.warning("aborted")
#    ramp(smu, 0, args.ramp_speed)
#    smu.write(':OUTP OFF\r')
#    sys.exit(0)
###----------------------------------------------------------------------------------------------------------------### 

ok_min = float(raw_input("what is the minimum voltage?[V]\n"))
ok_max = float(raw_input("what is the maximum voltage?[V]\n"))
ok_step = float(raw_input("what is the length between each measurement?[V]\n"))
ok_direction = raw_input("What is the direction (P/N)?\n")

voltages = np.arange(ok_min,ok_max+ok_step,ok_step)
if (ok_direction == 'N'):
    voltages = voltages*(-1)
logging.info(voltages)

###----------------------------------------------------------------------------------------------------------------### 
### sanity checks - just to be sure!
ok = raw_input("are you sure you want to continue? [Y/N]\n")
if ok != "Y":
    sys.exit(0)
    
try:  
    file_name='measurement_'+str(time.ctime())  
    with open(file_name+'.txt', 'w') as writer:
        writer.write('voltage[V],current[A]\n')
        smu.write(':OUTP ON\r')
        for i in voltages:
            logging.info("preparing for measurement(s) at " + str(i)+"V")
            ramp(smu, i, args.ramp_speed) ## ramp up to i V
            time.sleep(args.time_delay)
            for n in range(args.number_meas):
                logging.info("taking measurement")
                smu.write(':READ?\r')
                data = smu.readline()
                logging.info(data)
                writer.write(data)
                time.sleep(args.meas_speed)
    ramp(smu, 0, args.ramp_speed) ## ramp down to 0 V
    smu.write(':OUTP OFF\r')
    
except KeyboardInterrupt:
    logging.warning("aborted")
        
    ramp(smu, 0, args.ramp_speed)
    smu.write(':OUTP OFF\r')
    sys.exit(0)

#Plotting data and saving file    
data = pd.read_csv(file_name+str('.txt'))
var = np.split(data['current[A]'],len(voltages))
var_mean = np.mean(var, axis = 1)

fig,ax = plt.subplots()
ax.plot(voltages,var_mean,'-o')
ax.set(xlabel='Voltage [V]',ylabel='current [A]')
fig.savefig(file_name+'.png')


