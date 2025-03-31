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
simulation_period=4,
nat_freq_mean=40,
nat_freq_std=0,
GenerateRandom=False,
StimTstart=2,
StimTend=3,
StimWeigth=100,
StimFreq=13,
SEED=seed)

model.initializeForcingNodes(7)
model.setGlobalCoupling(360)
model.setMeanTimeDelay(0.021)
model.setRank(14)
model.setDesiredPoles([-200,-200,-200,-200,-200,-200,-200,
                       -200,-200,-200,-200,-200,-200,-200])
num_of_realizations=1


model.simulate()
dynamics=np.fliplr(model.x)
u=model.u_Out
# directory='/mnt/usb-Seagate_Basic_NABS42YJ-0:0-part2/MatricesStructure/'
# filename=directory+'Kuramoto_Controled_seed%d.mat'%(seed)
# data={'theta':dynamics,'u':u}
# sio.savemat(filename,data)  



#Multiprocessing
# for j in range(1):
#     print('Starting simulations')
#     lock = Lock()
#     with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
#         executor.map(RunKuramotoFor, itertools.product(K_Array, mean_delay_Array,[j]))
#%%
import scipy.signal as signal
x=dynamics
simulation_period=model.simulation_period
dt=model.dt
plt.figure()
plt.subplot(2,1,1)
f,Pxx=signal.welch(np.sin(x[:,::]),fs=1000,nperseg=2500,noverlap=1250)
plt.plot(f[0:200],Pxx.T[0:200,:],color='C0',alpha=0.4)
plt.plot(f[0:200],Pxx[7,0:200],color='C1',alpha=0.8)
plt.plot(f[0:200],np.mean(Pxx,axis=0)[0:200],'k',linewidth=2)
plt.xlabel('Frequency')
plt.ylabel('PSD')
plt.subplot(2,1,2)
t=np.arange(dt,simulation_period,dt)
plt.plot(t,u[7,:])
plt.ylabel('Control signal')
plt.xlabel('time (s)')
plt.tight_layout()
#%%
# x1=dynamics
# plt.subplot(2,1,2)
# f,Pxx1=signal.welch(np.sin(x1[:,::]),fs=1000,nperseg=1000,noverlap=500)
# plt.plot(f,Pxx1.T,color='C0',alpha=0.4)
# plt.plot(f,np.mean(Pxx1,axis=0),'k',linewidth=2)
# plt.xlabel('Frequency')
# plt.ylabel('PSD')

