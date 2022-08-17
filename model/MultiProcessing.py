from multiprocessing import Lock
from KuramotoClassFor import Kuramoto
import concurrent.futures
import itertools 
import gc
import numpy as np
import csv 
from npy_append_array import NpyAppendArray
import matplotlib.pyplot as plt
def RunKuramotoFor(param_tuple):
    model=Kuramoto()
    model.setGlobalCoupling(param_tuple[0])
    model.setMeanTimeDelay(param_tuple[1])
    num_of_realizations=1
    # for j in range(6):
    for i in range(num_of_realizations):
        R,R_Driving,R_Driven,Dynamics=model.simulate(Forced=True)
        # R,R_Driving,R_Driven=model.simulate(Forced=True)
        # model.plotEnsOrd(R_Driven)
        # model.plotEnsOrd(R_Driving)
        # model.plotEnsOrd(R)
        # plt.show()
        filename=r'/media/ahmed/VIBRAIN_WP1/EXP14/Dynamics'+' K='+ str(param_tuple[0])+' MD= '+str(param_tuple[1])+' No='+str(param_tuple[2])+'.csv'
        np.savetxt(filename, Dynamics, delimiter=',')
        with open(r'/media/ahmed/VIBRAIN_WP1/EXP14/Dynamics'+' K='+ str(param_tuple[0])+' MD= '+str(param_tuple[1])+' No='+str(param_tuple[2])+'.csv', "a") as f:
            writer = csv.writer(f)
        #     writer.writerow(Dynamics)
        with open(r'/media/ahmed/VIBRAIN_WP1/EXP14/General Realization'+' K='+ str(param_tuple[0])+' MD= '+str(param_tuple[1])+'.csv', "a") as f:
            writer = csv.writer(f)
            writer.writerow(R)
        with open(r'/media/ahmed/VIBRAIN_WP1/EXP14/Driving Realization'+' K='+ str(param_tuple[0])+' MD= '+str(param_tuple[1])+'.csv', "a") as f:
            writer = csv.writer(f)
            writer.writerow(R_Driving)
        with open(r'/media/ahmed/VIBRAIN_WP1/EXP14/Driven Realization'+' K='+ str(param_tuple[0])+' MD= '+str(param_tuple[1])+'.csv', "a") as f:
            writer = csv.writer(f)
            writer.writerow(R_Driven)      

        # with open(r'EXP10/General Realization'+' K='+ str(param_tuple[0])+' MD= '+str(param_tuple[1])+'.csv', "a") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(R)
        # with open(r'EXP10/Driving Realization'+' K='+ str(param_tuple[0])+' MD= '+str(param_tuple[1])+'.csv', "a") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(R_Driving)
        # with open(r'EXP10/Driven Realization'+' K='+ str(param_tuple[0])+' MD= '+str(param_tuple[1])+'.csv', "a") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(R_Driven)                                 

        del R,R_Driven,R_Driving,Dynamics
        gc.collect()
    del model
    gc.collect()

MD_Array=np.arange(0, 0.012, 0.001).tolist()
K_Array=np.arange(3, 12, 1).tolist()
# MD_Array=np.arange(0.008, 0.011, 0.001).tolist()
# # MD_Array=[0, 0.001, 0.0015 , 0.002]
# K_Array=np.arange(0.1, 1, 0.2).tolist()
# K_Array=[5]
#param_tuple=[5,0.007,100]
#RunKuramotoFor(param_tuple)
for j in range(1):

    lock = Lock()
    with concurrent.futures.ProcessPoolExecutor(max_workers=14) as executor:
        executor.map(RunKuramotoFor, itertools.product(K_Array, MD_Array,[j]))

