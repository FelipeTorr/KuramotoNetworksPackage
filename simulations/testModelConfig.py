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
import model.parserConfig as parser 

def RunKuramotoFor(configFile):
    time.sleep(0.1)
    model=Kuramoto()
    num_of_realizations=1
    parameters=parser.loadData(configFile)
    model.loadParameters(parameters)
    experiment=parameters['experiment_name']
    N=parameters['n_nodes']
    K=parameters['K']
    MD=parameters['mean_delay']
    seed=parameters['seed']
    # for j in range(6):
    for i in range(num_of_realizations):
        print('Simulating %s with %d nodes network using K=%.3f and MD=%.3f'%(experiment,N,K,MD))
        R,Dynamics=model.simulate(Forced=True)
        directory='../output_timeseries/'
        filename=directory+'%s_N%d_K%.3F_MD%.3f_seed%d.mat'%(experiment,N,K,MD,seed)
        data={'theta':Dynamics,'kop':R}
        sio.savemat(filename,data)  
        del R,Dynamics
        gc.collect()
    del model
    gc.collect()

if __name__=='__main__':
    config_directory='../input_config/multiple/'
    config_files=[]
    for file in os.listdir(config_directory):
        if file.split('.')[1]=='txt':
            config_files.append(config_directory+file)
            
    parameters=parser.loadData(config_files[0])
    print(parameters)
    max_workers=parameters['max_workers']
    
    #RunKuramotoFor(config_files[0])
    #Multiprocessing
    for j in range(1):
        print('Starting simulations')
        lock = Lock()
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            executor.map(RunKuramotoFor, config_files)
