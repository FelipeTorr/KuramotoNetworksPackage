from ast import Del
from symtable import Symbol
import networkx as nx
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import os
from tqdm import tqdm
from numpy import pi, random, max
from scipy import signal
# https://jitcdde.readthedocs.io/en/stable/
from jitcdde import jitcdde, y, t
from symengine import sin, Symbol
import symengine
import sympy
import glob
from PIL import Image
from sklearn.preprocessing import normalize
from math import comb
from numpy import linalg as LA
from matplotlib import colors
import gc
from math import floor
from multiprocessing import Lock
# from KuramotoClass import Kuramoto
import concurrent.futures
import itertools 
import gc
import io
from npy_append_array import NpyAppendArray

# from networkx.algorithms.community import k_clique_communities


class Kuramoto:
    def __init__(self,
                SC=None,
                K=5,
                dt=1.e-4,
                T=100,
                StimTstart=30,StimTend=60,StimFreq=40,StimAmp=300,
                n_nodes=90,natfreqs=None,GenerateRandom=True,SEED=20,
                MD=0.010):

        '''
        SC: is the adjacency matrix (SC).
        K: is the global coupling strength.
        T    : is the simulation time.
        dt: is the integration time step.
        StimFreq: is the stimulation frequency. 
        StimAmp: is the stimulation Amplitude.
        StimTstart: is the onset start time.
        StimTend  : is the offset time.
        n_nodes: is the number of nodes.
        natfreqs: is the natural frequencies of the nodes. 
        Delay: is the delay matrix.
        MD: is the mean delay (in Cabral, this delay is a scale like global coupling strength)
        GenerateRandom: random natural frequencies every time if True, if False
        SEED: guarantees that the natfreqs are the same at each run
        
        '''
        if n_nodes is None and natfreqs is None:
            raise ValueError("n_nodes or natfreqs must be specified")
        self.n_nodes=n_nodes
        self.MD=MD
        self.dt=dt
        self.T=T
        self.K=K
        self.param_sym_func=symengine.Function('param_sym_func')
        self.StimTstart=StimTstart
        self.StimTend=StimTend
        self.StimFreq=StimFreq*2*np.pi
        self.StimAmp=StimAmp
        self.ForcingNodes=np.zeros((self.n_nodes,1))
        self.ForcingNodes[80:90,:]=1

        self.Forced=False        

        self.SC=self.initializeStructuralConnectivity()
        # self.initializeArbitraryGraph()
        self.mean_nat_freq=40
        self.variance=0
        
        self.natfreqs=self.initializeNatFreq(natfreqs,SEED,n_nodes,GenerateRandom) # This initialize the natfreq
        self.ω = 2*np.pi*self.natfreqs*np.ones(n_nodes)

        self.Delay=self.initializeTimeDelays()
        self.act_mat=None # When the simulation run, this value is updated with the dynamics. 

    def initializeNatFreq(self,natfreqs,SEED,n_nodes,GenerateRandom):
        if natfreqs is not None:
            natfreqs=natfreqs
            n_nodes=len(natfreqs)
        else:
            self.n_nodes=n_nodes
            if GenerateRandom==True:
                natfreqs=np.random.normal( self.mean_nat_freq,self.variance ,size=self.n_nodes)
            else:
                np.random.seed(SEED)
                natfreqs=np.random.normal( self.mean_nat_freq,self.variance ,size=self.n_nodes)
        return natfreqs

    def initializeTimeDelays(self):
        No_nodes=self.n_nodes
        D = loadmat('./AAL_matrices.mat')['D']
        D=D[-No_nodes:,-No_nodes:]
        C=self.SC
        MD=self.MD
        if No_nodes>90:
            print('Max No. Nodes is 90')
            No_nodes=90
        D /= 1000 # Distance matrix in meters

        if MD==0:
            τ = np.zeros_like(C) # Set all delays to 0.
        else:
            τ = D / D[C>0].mean() * MD 
            # τ = τ.astype(np.int)
        τ[C==0] = 0.
        self.Delay=τ
        return τ



    def initializeArbitraryGraph(self):
        # Initialize erdos renyi graph
        mu=0.05
        sigma=0.2*mu
        n=self.n_nodes
        np.random.seed(2)
        C = np.random.normal(mu, sigma, (n,n))
        indices = np.random.choice(np.arange(C.shape[0]), replace=False,size=int(C.shape[0]*0.4 ))
        C[indices] = 0
        indices = np.random.choice(np.arange(C.shape[0]), replace=False,size=int(C.shape[0]*0.3 ))
        C[indices] = -C[indices]
        C = (C + C.T)/2
        
        C[np.diag(np.ones(n))==0] /= C[np.diag(np.ones(n))==0].mean()
        np.fill_diagonal(C, 0)
        n_zeros = np.count_nonzero(C==0)
        # print(n_zeros)
        
        

        # D = loadmat('./AAL_matrices.mat')['D']
        MD=self.MD
        VD=MD*0.2
        D= np.random.normal(MD, VD, (n,n))
        D = (D + D.T)/2
        D=D[:n,:n]
        np.fill_diagonal(D, 0)
        if n>90:
            print('Max No. Nodes is 90')
            n=90
        if MD==0:
            τ = np.zeros_like(C) # Set all delays to 0.
        else:
            τ = D / D[C>0].mean() * MD 
            # τ = τ.astype(np.int)
        τ=D
        τ[C==0] = 0.
        self.Delay=τ
        self.SC=C
        # print(C)
        # print(τ)
        return C,τ


    def initializeStructuralConnectivity(self):
        n=self.n_nodes
        
        C = loadmat('./AAL_matrices.mat')['C']
        if n>90:
            print('Max No. Nodes is 90')
            n=90
        C=C[-n:,-n:]
        C[np.diag(np.ones(n))==0] /= C[np.diag(np.ones(n))==0].mean()

        return C
    def setMeanTimeDelay(self, MD):
        self.MD=MD
        self.initializeTimeDelays()
    
    def setGlobalCoupling(self,K):
        self.K=K

    def param(self,y,arg):
        '''
        This function present the stimulation from Tstart to Tend 
        It is used below in the integration function, 
        t is here because of the default parameters of the odeint function.
        '''

        if self.StimTend>self.T:
            print("Tend Cannot be larger than the simulation time which is"+'%.1f' %self.T)
        if arg<self.StimTstart:
            return 0
        elif arg>self.StimTstart and arg<self.StimTend: 
            return 1
        else:
            return 0

    def kuramotos(self):
        for i in range(self.n_nodes):
            yield self.ω[i] +self.K*sum(
                self.SC[j, i] * sin( y(j , t - (self.Delay[i, j]) ) - y(i) )
                for j in range(self.n_nodes)
            )

    def kuramotosForced(self):
        Delta=self.ForcingNodes
        for i in range(self.n_nodes):
            yield self.ω[i] + Delta[i,0]*self.param_sym_func(t)*self.StimAmp*sin(self.StimFreq*t-y(i))+self.K*sum(
                self.SC[j, i] * sin( y(j , t - (self.Delay[i, j]) ) - y(i) )
                for j in range(self.n_nodes)
            )


    def IntegrateDD(self):
        T=self.T
        dt=self.dt
        τ=self.Delay
        n=self.n_nodes
        if self.Forced:
            DDE = jitcdde(self.kuramotosForced, n=n, verbose=False,callback_functions=[( self.param_sym_func, self.param,1)])
            DDE.compile_C(simplify=False, do_cse=False, chunk_size=150)
            DDE.set_integration_parameters(rtol=1e-5, atol=1e-5)
            DDE.constant_past(random.uniform(0, 2*np.pi, n), time=0.0)
            DDE.integrate_blindly(max(τ), 0.00001)

        elif not self.Forced:
            DDE = jitcdde(self.kuramotos, n=n, verbose=True)
            DDE.compile_C(simplify=False, do_cse=False, chunk_size=150)
            DDE.set_integration_parameters(rtol=1e-5, atol=1e-5)
            DDE.constant_past(random.uniform(0, 2*np.pi, n), time=0.0)
            DDE.integrate_blindly(max(τ), 0.001)

        output = []
        for time in tqdm(DDE.t + np.arange(0, T,dt )):
            output.append([*DDE.integrate(time)])
        output=np.asarray(output)
        
        
        del DDE
        gc.collect()
        return output

    
    def simulate(self,Forced):
        self.Forced=Forced
        Dynamics=self.IntegrateDD()
        forcing_indices=(np.where(self.ForcingNodes==1))[0]
        driven_indices=(np.where(self.ForcingNodes==0))[0]
        # print(forcing_indices)
        # print(driven_indices)
        R_Driving=self.calculateOrderParameter(Dynamics[:,forcing_indices])
        R_Driven=self.calculateOrderParameter(Dynamics[:,driven_indices])
        R=self.calculateOrderParameter(Dynamics)
        # del Dynamics
        # gc.collect()
        return R,R_Driving,R_Driven,Dynamics

    def testIntegrateForced(self):
        R=self.simulate(Forced=True)
        self.plotOrderParameter(R)

        
    def testIntegrate(self):

        R=self.simulate(Forced=True)
        self.plotOrderParameter(R)
        # OP=OrderParameter(output)
        # Step=10 # This is for the frames in GIF, If decreased --> more frames Ex. T-10 , Step= 10, dt=10^-3 --> Thus 1000 frame 
        
        # animateSync(T,dt,Step,Dynamics) # Ram Warning, This saves each frame -> GIF -> Deleted the images. 

        return R
    @staticmethod
    def phase_coherence(angles_vec):

        '''
        Compute global order parametr R_t - mean length of resultant vector
        re^(i*epsi)=(1/N)* sum ( e^(i*theta_m))
        '''
        suma=sum([np.e ** (1j*i) for i in angles_vec ])
        return abs(suma/len(angles_vec))

    def calculateOrderParameter(self,act_mat):
        R=[self.phase_coherence(vec) for vec in act_mat]
        # plt.plot(R)
        # plt.ylabel('Order parameter at'+str(self.K)+str(self.MD), fontsize=25)
        # plt.title(r'$<T>=$'+'%.001f ' % self.MD+ r'$  <K>=$'+'%.1f ' % self.K)
        # plt.xlabel('Time', fontsize=25)
        # plt.ylim((-0.01, 1))
        # plt.savefig('OP vs Time'+str(self.K)+str(self.MD)+'.png')
        # # self.plotOnOffSet()
        # plt.show()
        # gc.collect()
        del act_mat
        gc.collect()
        return R


    def plotOrderParameter(self,act_mat):
        R=[self.phase_coherence(vec) for vec in act_mat]
        plt.figure(figsize=(12,4))
        plt.plot(R)
        plt.ylabel('Order parameter at'+str(self.K)+str(self.MD), fontsize=25)
        plt.title(r'$<T>=$'+'%.001f ' % self.MD+ r'$  <K>=$'+'%.1f ' % self.K)
        plt.xlabel('Time', fontsize=25)
        plt.ylim((-0.01, 1))
        plt.savefig('OP vs Time'+str(self.K)+str(self.MD)+'.png')
        self.plotOnOffSet()
        # x=np.random.random()
        np.save('Ord'+' K= '+str(self.K)+'  MD= '+str(self.MD)+' .npy',R)
        # R_mean=np.mean(R[m:len(R)])
        del R
        gc.collect()
        plt.show()

    def plotOnOffSet(self):
        plt.axvline(self.StimTstart/self.dt,color='r',label='Onset',linestyle='--')
        plt.axvline(self.StimTend/self.dt,color='b',label='Offset',linestyle='--')
        plt.legend()

    def Animation(self,fp_in,name):
        fp_out = name+".gif"

        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img, *imgs = [Image.open(f) for f in fp_in]
        img.save(fp=fp_out, format='GIF', append_images=imgs,
                    save_all=True, duration=150, loop=0)
        self.remove_img()

    def remove_img(self):

        [os.remove(file) for file in os.listdir() if file.endswith('.png')]
    def plotPhaseInteraction(self,act_mat=None):
        if (act_mat==None).all:
            act_mat=self.SC
        plt.figure(figsize=(12,4))
        plt.plot(np.sin(act_mat.T))
        plt.xlabel('Time In Steps of dt')
        plt.ylabel(r'$\sin(\theta)$')
        plt.savefig('Phase Synchrony.png')
        plt.show()


########################################################################
#Forced Kuramoto Model
########################################################################

    def plotEnsOrd(self,EnsambleAverage):
        plt.figure(figsize=(12,4))
        plt.plot(EnsambleAverage)
        plt.ylabel('Order parameter at'+str(self.K)+str(self.MD), fontsize=25)
        plt.title(r'$<T>=$'+'%.001f ' % self.MD+ r'$  <K>=$'+'%.1f ' % self.K)
        plt.xlabel('Time', fontsize=25)
        plt.ylim((-0.01, 1))
        
        plt.axvline(self.StimTstart/self.dt,color='r',label='Onset',linestyle='--')
        plt.axvline(self.StimTend/self.dt,color='b',label='Offset',linestyle='--')
        plt.legend()
        plt.savefig('EnsembleAverage K='+str(self.K)+' MD='+str(self.MD)+'.png')
        plt.close()               
    


    def plotEnsemble(self,MD_Array,K_Array):
        n=len(K_Array)
        m=len(MD_Array)
        dt=self.dt
        StimTstart=self.StimTstart
        StimTend=self.StimTend
        for i in range(n):
            for j in range(m):
                Ord=np.load('EnsembleAverage'+'K='+ str(K_Array[i])+' MD= '+str(MD_Array[j])+'.npy')
                plt.figure(figsize=(12,4))
                plt.plot(Ord)
                plt.xlabel('Time', fontsize=25)
                plt.ylim((-0.01, 1))
                plt.title('EnsembleAverage'+'K='+ str(K_Array[i])+' MD= '+str(MD_Array[j]))
                plt.axvline(StimTstart/dt,color='r',label='Onset',linestyle='--')
                plt.axvline(StimTend/dt,color='b',label='Offset',linestyle='--')
                plt.legend()
                plt.savefig('EnsembleAverage'+'K='+ str(K_Array[i])+' MD= '+str(MD_Array[j])+'.png')


    def plotEnsembleAv(self,param_tuple):
        num_of_realizations=550
        
        #Rensamble=np.zeros((int(self.T/self.dt),num_of_realizations))
        Rensamble=np.zeros((250000,num_of_realizations))
        for i in range(num_of_realizations):

            R=np.load('../../Var/EXP2.6/OrderParameter'+' K='+ str(param_tuple[0])+' MD= '+str(param_tuple[1])+' Realization '+str(i)+'.npy')
            #print(i)
            # R=np.load('OrderParameter'+' K='+ str(param_tuple[0])+' MD= '+str(param_tuple[1])+' Realization '+str(i)+'.npy')
            Rensamble[:,i]=R[250000:500000]
            del R
            gc.collect()
        Ord=Rensamble.mean(1)
        plt.figure(figsize=(12,4))
        plt.plot(Ord)
        plt.xlabel('Time', fontsize=25)
        plt.ylim((-0.01, 1))
        plt.title('EnsembleAverage'+'K='+ str(param_tuple[0])+' MD= '+str(param_tuple[1]))
        #plt.axvline(self.StimTstart/self.dt,color='r',label='Onset',linestyle='--')
        #plt.axvline(self.StimTend/self.dt,color='b',label='Offset',linestyle='--')
        plt.axvline(self.StimTstart/self.dt-250000,color='r',label='Onset',linestyle='--')
        plt.axvline(self.StimTend/self.dt-250000,color='b',label='Offset',linestyle='--')
        plt.legend()
        plt.savefig('EnsembleAverage'+'K='+ str(param_tuple[0])+' MD= '+str(param_tuple[1])+'.png')

    def plotAllEnsebles(self,MD_Array,K_Array):
        lock = Lock()
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            executor.map(self.plotEnsembleAv, itertools.product(K_Array, MD_Array))
   
    # def multiProcForced(self,MD_Array,K_Array):
    #     lock = Lock()
    #     with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    #         executor.map(self.RunKuramotoFor, itertools.product(K_Array, MD_Array))



# MD_Array=np.arange(0.012, 0.020, 0.002).tolist()
# K_Array=np.arange(6, 8, 1).tolist()
#MD_Array=[0.008]
#K_Array=[6]
# model=Kuramoto()
# model.multiProcForced(MD_Array,K_Array)
#model.plotAllEnsebles(MD_Array,K_Array)





