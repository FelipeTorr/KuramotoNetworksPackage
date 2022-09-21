#!/usr/bin/python
# -*- coding: utf-8 -*-
#from ast import Del
from symtable import Symbol
from scipy.io import loadmat
import numpy as np
import os
import sys
sys.path.append(os.path.abspath('../'))
try:
    import analysis.connectivityMatrices as matrices
except ModuleNotFoundError:
    import KuramotoNetworksPackage.analysis.connectivityMatrices as matrices
from tqdm import tqdm
from numpy import pi, random, max
from scipy import signal
# https://jitcdde.readthedocs.io/en/stable/
from jitcdde import jitcdde, y, t
from symengine import sin, Symbol
import symengine
import sympy
import gc

# from KuramotoClass import Kuramoto
import time as tm

# from networkx.algorithms.community import k_clique_communities


class Kuramoto:
    def __init__(self,
                struct_connectivity=None,
                delays_matrix=None,
                K=5,
                dt=1.e-4,
                simulation_period=100,
                StimTstart=0,StimTend=0,StimFreq=0,StimAmp=0,
                n_nodes=4,natfreqs=None,
                nat_freq_mean=0,nat_freq_std=2,
                GenerateRandom=True,SEED=2,
                mean_delay=0.10):

        '''
        struct_connectivity: is the adjacency matrix (struct_connectivity).
        K: is the global coupling strength.
        simulation_period    : is the simulation time.
        dt: is the integration time step.
        StimFreq: is the stimulation frequency. 
        StimAmp: is the stimulation Amplitude.
        StimTstart: is the onset start time.
        StimTend  : is the offset time.
        n_nodes: is the number of nodes.
        natfreqs: is the natural frequencies of the nodes. 
        delays_matrix: is the delay matrix.
        mean_delay: is the mean delay (in Cabral, this delay is a scale like global coupling strength)
        random_nat_freq: random natural frequencies every time if True, if False
        SEED: guarantees that the natfreqs are the same at each run
        
        '''
        if n_nodes is None and natfreqs is None:
            raise ValueError("n_nodes or natfreqs must be specified")
        self.n_nodes=n_nodes
        self.mean_delay=mean_delay
        self.dt=dt
        self.simulation_period=simulation_period
        self.K=K
        self.param_sym_func=symengine.Function('param_sym_func')
        self.StimTstart=StimTstart
        self.StimTend=StimTend
        self.StimFreq=StimFreq*2*np.pi
        self.StimAmp=StimAmp
        self.ForcingNodes=np.zeros((self.n_nodes,1))
        #self.ForcingNodes[80:90,:]=0
        self.mean_nat_freq=nat_freq_mean
        self.std_nat_freq=nat_freq_std
        self.seed=SEED
        self.random_nat_freq=GenerateRandom

        if struct_connectivity is None:
            self.struct_connectivity=self.load_struct_connectivity()
        else:
            self.struct_connectivity=struct_connectivity
            self.n_nodes=len(struct_connectivity)
        
        if delays_matrix is None:
            self.delays_matrix=self.initializeTimeDelays()
        else:
            assert np.shape(delays_matrix)[0]==np.shape(delays_matrix)[1], 'Delays must be a square matrix' 
            assert np.shape(delays_matrix)[0]==np.shape(struct_connectivity)[0], 'SC and Delays matrix must be of the same size' 
            self.delays_matrix=delays_matrix
        self.applyMean_Delay()
        
        self.natfreqs=self.initializeNatFreq(natfreqs) # This initialize the natfreq
        self.ω = 2*np.pi*self.natfreqs
        self.global_coupling=self.K/self.n_nodes
        
        self.act_mat=None # When the simulation run, this value is updated with the dynamics. 
    
    def applyMean_Delay(self):
        #Apply a scaling factor to the delays matrix in order to obtain the specified mean delay
        if self.mean_delay==0:
            self.delays_matrix=np.zeros((self.n_nodes,self.n_nodes))
        else:
            self.delays_matrix=self.delays_matrix/np.mean(self.delays_matrix[self.struct_connectivity>0])*self.mean_delay
    
    def initializeNatFreq(self,natfreqs):
        #Set the natural(intrinsic) frequencies of the oscillators
        
        if natfreqs is not None:
            if type(natfreqs)==int or type(natfreqs)==float or type(natfreqs)==np.float32:
                #Equal frequency for all nodes
                natfreqs=natfreqs*np.ones(self.n_nodes) 
            elif self.n_nodes == len(natfreqs):
                #Frequencies specified in an array of the same length than n_nodes
                natfreqs=natfreqs
            elif type(natfreqs)==str:
                natfreqs=loadmat(natfreqs)['natfreqs']
                if self.n_nodes == len(natfreqs):
                    #Frequencies specified in an array of the same length than n_nodes
                    natfreqs=natfreqs
                else:
                    print('Natural frequencies are bad defined')
            else:
                print('Natural frequencies are bad defined')
        else:
            #Generate random natural frequencies from a Gaussian distribution
            if self.random_nat_freq==True:
                natfreqs=np.random.normal( self.mean_nat_freq,self.std_nat_freq ,size=self.n_nodes)
            else:
                np.random.seed(self.seed)
                natfreqs=np.random.normal( self.mean_nat_freq,self.std_nat_freq ,size=self.n_nodes)
        return natfreqs

    
    def loadParameters(self,parameters):
        self.K=parameters['K']
        self.n_nodes=parameters['n_nodes']
        self.mean_delay=parameters['mean_delay']
        self.dt=parameters['dt']
        self.simulation_period=parameters['simulation_period']
        self.StimTstart=parameters['StimTstart']
        self.StimTend=parameters['StimTend']
        self.StimAmp=parameters['StimAmp']
        self.StimFreq=parameters['StimFreq']
        self.seed=parameters['seed']
        self.nat_freq_mean=parameters['nat_freq_mean']
        self.nat_freq_std=parameters['nat_freq_std']
        self.random_nat_freq=parameters['random_nat_freq']
        self.natfreqs=self.initializeNatFreq(parameters['nat_freqs'])
        self.struct_connectivity=matrices.loadConnectome(self.n_nodes,parameters['struct_connectivity'])
        self.delays_matrix=matrices.loadDelays(self.n_nodes,parameters['delay_matrix'])
        self.applyMean_Delay()
        self.ω = 2*np.pi*self.natfreqs
        self.ForcingNodes=np.zeros((self.n_nodes,1))
        self.global_coupling=self.K/self.n_nodes


    def initializeTimeDelays(self):
        No_nodes=self.n_nodes
        D = loadmat('../input_data/AAL_matrices.mat')['D']
        D=D[-No_nodes:,-No_nodes:]
        C=self.struct_connectivity
        mean_delay=self.mean_delay
        if No_nodes>90:
            print('Max No. Nodes is 90')
            No_nodes=90
        D /= 1000 # Distance matrix in meters
        self.delays_matrix=D
        return D

    def load_struct_connectivity(self):
        n=self.n_nodes
        
        C = loadmat('../input_data/AAL_matrices.mat')['C']
        if n>90:
            print('Max No. Nodes is 90')
            n=90
        C=C[-n:,-n:]
        C[np.diag(np.ones(n))==0] /= C[np.diag(np.ones(n))==0].mean()

        return C
    def setMeanTimeDelay(self, mean_delay):
        self.mean_delay=mean_delay
        self.applyMean_Delay()

    def setGlobalCoupling(self,K):
        self.K=K
        self.global_coupling=self.K/self.n_nodes

    def param(self,y,arg):
        '''
        This function present the stimulation from Tstart to Tend 
        It is used below in the integration function, 
        t is here because of the default parameters of the odeint function.
        '''

        if self.StimTend>self.simulation_period:
            print("Tend Cannot be larger than the simulation time which is"+'%.1f' %self.T)
        if arg<self.StimTstart:
            return 0
        elif arg>self.StimTstart and arg<self.StimTend: 
            return 1
        else:
            return 0
    def initial_phases(self):
        return 2*np.pi*np.random.random(size=self.n_nodes)
    
    def kuramotosZero(self,y,t):
        ϴ_i,ϴ_j=np.meshgrid(y,y)
        dphi_dt= self.ω + (self.global_coupling)*( self.struct_connectivity * np.sin( ϴ_j - ϴ_i ) ).sum(axis=0)
        return dphi_dt

    def kuramotos(self):
        for i in range(self.n_nodes):
            yield self.ω[i] +self.global_coupling*sum(
                self.struct_connectivity[i, j] * sin( y(j,t - (self.delays_matrix[i, j])) - y(i))
                for j in range(self.n_nodes)
            )

    def kuramotosForced(self):
        Delta=self.ForcingNodes
        for i in range(self.n_nodes):
            yield self.ω[i] + Delta[i,0]*self.param_sym_func(t)*self.StimAmp*sin(self.StimFreq*t-y(i))+self.global_coupling*sum(
                self.struct_connectivity[i, j] * sin( y(j , t - (self.delays_matrix[i, j])) - y(i))
                for j in range(self.n_nodes)
            )


    def IntegrateDD(self):
        simulation_period=self.simulation_period
        dt=self.dt
        max_delay=np.max(self.delays_matrix)
        n=self.n_nodes
        if n<20:
            chunksize=20
        elif n<360:
            chunksize=4
        else:
            chunksize=1
        time_start=tm.time()
        if self.Forced:
            DDE = jitcdde(self.kuramotosForced, n=n, verbose=False,delays=self.delays_matrix.flatten(),callback_functions=[( self.param_sym_func, self.param,1)])
            DDE.compile_C(simplify=False, do_cse=False, chunk_size=chunksize,verbose=False)
            DDE.set_integration_parameters(rtol=1e-10, atol=1e-5,pws_max_iterations=1)
            DDE.constant_past(random.uniform(0, 2*np.pi, n), time=0.0)
            if max_delay>0:
                DDE.integrate_blindly(max_delay, dt*1.0e-2)
        
        elif not self.Forced:
            DDE = jitcdde(self.kuramotos, n=n,delays=self.delays_matrix.flatten(),verbose=False)
            DDE.compile_C(simplify=False, do_cse=False, chunk_size=chunksize,verbose=False)
            DDE.set_integration_parameters(rtol=1e-10, atol=1e-5,pws_max_iterations=1)
            DDE.constant_past(random.uniform(0, 2*np.pi, n), time=0.0)
            if max_delay>0:
                DDE.integrate_blindly(max_delay, dt)

        #DDE.step_on_discontinuities()
        time_end=tm.time()-time_start
        print('Compiled in %.4f seconds'%time_end)
        output = []
        for time in tqdm(DDE.t + np.arange(0, simulation_period,dt )):
            output.append([*DDE.integrate(time)])
        output=np.asarray(output)
        del DDE
        gc.collect()
        return output
        
        #else:
        #    ϴ_o=self.initial_phases()

        #    t = np.linspace(0, self.simulation_period, int(self.simulation_period/self.dt))
        #    dynamics = odeint(self.kuramotosZero, ϴ_o, t)
            
        #    return dynamics


        
        
    def interact(self,K):
        import matplotlib.pyplot as plt
        self.K=K
        R,Dynamics=self.simulate()
        t=np.linspace(0,self.simulation_period,int(self.simulation_period//self.dt)+1)
        plt.figure(figsize=(12,4))
        plt.plot(t,R)
        plt.show()

    
    def simulate(self,Forced=False):
        self.Forced=Forced
        Dynamics=self.IntegrateDD()
        R=self.calculateOrderParameter(Dynamics)
        # del Dynamics
        # gc.collect()
        return R,Dynamics

    def testIntegrateForced(self):
        R=self.simulate(Forced=True)
        self.plotOrderParameter(R)

        
    def testIntegrate(self):

        R=self.simulate(Forced=True)
        self.plotOrderParameter(R)
        # OP=OrderParameter(output)
        # Step=10 # This is for the frames in GIF, If decreased --> more frames Ex. T-10 , Step= 10, dt=10^-3 --> Thus 1000 frame 
        
        # animateSync(simulation_period,dt,Step,Dynamics) # Ram Warning, This saves each frame -> GIF -> Deleted the images. 

        return R
    @staticmethod
    def phase_coherence(angles_vec):

        '''
        Compute global order parametr R_t - mean length of resultant vector
        re^(i*epsi)=(1/N)* sum ( e^(i*theta_m))
        '''
        suma=sum([np.exp(1j*i) for i in angles_vec ])
        return abs(suma/len(angles_vec))

    def calculateOrderParameter(self,act_mat):
        R=[self.phase_coherence(vec) for vec in act_mat]
        # plt.plot(R)
        # plt.ylabel('Order parameter at'+str(self.K)+str(self.mean_delay), fontsize=25)
        # plt.title(r'$<T>=$'+'%.001f ' % self.mean_delay+ r'$  <K>=$'+'%.1f ' % self.K)
        # plt.xlabel('Time', fontsize=25)
        # plt.ylim((-0.01, 1))
        # plt.savefig('OP vs Time'+str(self.K)+str(self.mean_delay)+'.png')
        # # self.plotOnOffSet()
        # plt.show()
        # gc.collect()
        del act_mat
        gc.collect()
        return R





