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
def RunKuramotoFor(param_tuple):
    
    #fully connected graph for Structural Connectivty
    #Homogeneous delay
    N=16
    τ=np.ones(N)
    τ=τ-np.diag(τ)
    
    seed=2
    
    model=Kuramoto(n_nodes=N,
    struct_connectivity=τ,
    delays_matrix=τ,
    simulation_period=10,
    nat_freq_mean=40,
    nat_freq_std=0,
    GenerateRandom=False,
    SEED=seed)
    
    model.setGlobalCoupling(param_tuple[0])
    model.setMeanTimeDelay(param_tuple[1])
    num_of_realizations=1
    
    # for j in range(6):
    for i in range(num_of_realizations):
        print('Simulating fully-connected %d nodes network using K=%.3f and MD=%.3f'%(N,param_tuple[0],param_tuple[1]))
        R,Dynamics=model.simulate(Forced=True)
        directory='../output_timeseries/'
        filename=directory+'testModel_N%d_K%.3F_MD%.3f_seed%d.mat'%(N,param_tuple[0],param_tuple[1],seed)
        data={'theta':Dynamics,'kop':R}
        sio.savemat(filename,data)  
        del R,Dynamics
        gc.collect()
    del model
    gc.collect()

mean_delay_Array=np.arange(0,0.2,0.1).tolist()
K_Array=np.arange(0, 2, 1.0).tolist()
#Single process
#param_tuple=[K_Array[0],mean_delay_Array[0],1]
#RunKuramotoFor(param_tuple)

#Multiprocessing
for j in range(1):
    print('Starting simulations')
    lock = Lock()
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        executor.map(RunKuramotoFor, itertools.product(K_Array, mean_delay_Array,[j]))

#Load and plot stored data
N=16
seed=2

directory='../output_timeseries/'
for K in K_Array:
    for MD in mean_delay_Array: 
        filename=directory+'testModel_N%d_K%.3f_MD%.3f_seed%d.mat'%(N,K,MD,2)
        file_dict=sio.loadmat(filename)
        theta=file_dict['theta']
        kop=file_dict['kop'][0]
        plt.plot(kop,label='K=%.3f MD=%.3f'%(K,MD))
plt.legend()
plt.show()