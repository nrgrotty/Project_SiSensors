import logging
import argparse
import serial
import sys
import time
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
### sanity checks - just to be sure!
ok = raw_input("are you sure you want to continue? [Y/N]\n")
if ok != "Y":
    sys.exit(0)
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
smu.write(':SENS:CURR:PROT 2E-8\r')
smu.write(':SENS:CURR:RANG MIN\r')
smu.write(':SENS:CURR:NPLC %s\r' % args.NPLC)
if args.rear_terms:
    smu.write(':ROUT:TERM REAR\r')
smu.write(':FORM:ELEM VOLT, CURR\r')
###----------------------------------------------------------------------------------------------------------------### 
### set voltage and take current measurements
try:
    smu.write(':OUTP ON\r')
    logging.info("preparing for measurement(s) at 10 V")
    ramp(smu, 10, args.ramp_speed) ## ramp up to 10 V
    time.sleep(args.time_delay)
    for n in range(args.number_meas):
        logging.info("taking measurement")
        smu.write(':READ?\r')
        data = smu.readline()
        logging.info(data)
        time.sleep(args.meas_speed)
    ramp(smu, 0, args.ramp_speed) ## ramp down to 0 V
    smu.write(':OUTP OFF\r')

except KeyboardInterrupt:
    logging.warning("aborted")
    ramp(smu, 0, args.ramp_speed)
    smu.write(':OUTP OFF\r')
    sys.exit(0)
###----------------------------------------------------------------------------------------------------------------### 
