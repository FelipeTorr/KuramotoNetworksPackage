#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:34:49 2024

@author: felipe
"""

import os
import sys
sys.path.append(os.path.abspath('../'))
from multiprocessing import Lock
from model.KuramotoControlled import Kuramoto
import concurrent.futures
import itertools 
import gc
import numpy as np
import scipy.io as sio
import csv 
import matplotlib.pyplot as plt
import time


time.sleep(0.1)
#fully connected graph for Structural Connectivty
#Homogeneous delay
N=90

seed=2

model=Kuramoto(n_nodes=N,
dt=1e-3,
simulation_period=30,
nat_freq_mean=40,
nat_freq_std=0,
GenerateRandom=False,
SEED=seed)

model.setGlobalCoupling(360)
model.setMeanTimeDelay(0.021)
model.setRank(12)
model.setDesiredPoles([-100,-100,-100,-100,-100,-100,
                       -100,-100,-100,-100,-100,-100])
num_of_realizations=1


model.simulate()
dynamics=np.fliplr(model.x)
directory='/mnt/usb-Seagate_Basic_NABS42YJ-0:0-part2/MatricesStructure/'
filename=directory+'Kuramoto_Controled_seed%d.mat'%(seed)
data={'theta':dynamics}
sio.savemat(filename,data)  



#Multiprocessing
# for j in range(1):
#     print('Starting simulations')
#     lock = Lock()
#     with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
#         executor.map(RunKuramotoFor, itertools.product(K_Array, mean_delay_Array,[j]))
#%%
import scipy.signal as signal
x=np.fliplr(dynamics)
f,Pxx=signal.welch(np.sin(x[:,10000::]),fs=1000,nperseg=5000,noverlap=2500)
plt.plot(f,np.mean(Pxx,axis=0))

