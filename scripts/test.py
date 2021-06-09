#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 10:08:09 2021

@author: neus
"""
import h5py 
with = h5py.File('signaldata.h5','r')
F.keys()
# pedestals = F['header/pedestal']

# ped = pedestals[0]


import h5py
filename = "file.hdf5"

with h5py.File(filename, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])