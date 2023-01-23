import os
import sys
sys.path.append(os.path.abspath('../'))
from multiprocessing import Lock
from model.KuramotoClassFor import Kuramoto
import concurrent.futures
import itertools 
import gc
import numpy as np
import scipy.io as sio
import csv 
from npy_append_array import NpyAppendArray
import matplotlib.pyplot as plt
import time 
def RunKuramotoFor(param_tuple):
    time.sleep(0.1)
    #fully connected graph for Structural Connectivty
    #Homogeneous delay
    N=8
    seed=2
    τ=np.ones(N)
    τ=τ-np.diag(τ)
    np.random.seed(seed)
    Cbig=np.random.rand(32,32)*0.01
    Cbig=Cbig-np.diag(Cbig)
    Cbig=np.abs(Cbig)
    n=5
    for j in range(2**n):    
        sum_row=np.sum(Cbig[j,:2**n])
        if sum_row<1e-6:
            Cbig[j,:]=Cbig[j,:]+0.1
            Cbig[j,j]=0
            sum_row=np.sum(Cbig[j,:2**n])
        Cbig[j,:]=Cbig[j,0:2**n]/sum_row*(2**n-1)
    C=Cbig[0:N,0:N]
    model=Kuramoto(n_nodes=N,
    struct_connectivity=τ,
    delays_matrix=τ,
    simulation_period=40,
    nat_freq_mean=40,
    nat_freq_std=1,
    GenerateRandom=False,
    dt=1e-3,
    SEED=seed)
    
    model.setGlobalCoupling(param_tuple[0])
    model.setMeanTimeDelay(param_tuple[1])
    num_of_realizations=1
    
    # for j in range(6):
    for i in range(num_of_realizations):
        print('Simulating fully-connected %d nodes network using K=%.3f and MD=%.4f'%(N,param_tuple[0],param_tuple[1]))
        R,Dynamics=model.simulate(Forced=True)
        directory='../output_timeseries/'
        filename=directory+'SIPAIM_Homogeneous_N%d_K%.3F_MD%.4f_seed%d.mat'%(N,param_tuple[0],param_tuple[1],seed)
        data={'theta':Dynamics,'kop':R}
        sio.savemat(filename,data)  
        del R,Dynamics
        gc.collect()
    del model
    gc.collect()

mean_delay_Array=[0.0]#np.arange(0.0005,0.02,0.0005).tolist()
K_Array=[20.0]
#Single process
#param_tuple=[K_Array[0],mean_delay_Array[0],1]
#RunKuramotoFor(param_tuple)

#Multiprocessing
for j in range(1):
    print('Starting simulations')
    lock = Lock()
    with concurrent.futures.ProcessPoolExecutor(max_workers=13) as executor:
        executor.map(RunKuramotoFor, itertools.product(K_Array, mean_delay_Array,[j]))

