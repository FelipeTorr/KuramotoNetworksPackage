#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 12:09:54 2024

@author: felipe
"""
from scipy.io import loadmat
import numpy as np
import os
import sys
sys.path.append(os.path.abspath('../'))
try:
    import analysis.connectivityMatrices as matrices
    import analysis.dynamicModes as dmd
    import analysis.control as control
except ModuleNotFoundError:
    import KuramotoNetworksPackage.analysis.connectivityMatrices as matrices
    import KuramotoNetworksPackage.analysis.dynamicModes as dmd
    import KuramotoNetworksPackage.analysis.control as control
from tqdm import tqdm
from numpy import pi, random, max
from scipy import signal


class Kuramoto:
    """
    Kuramoto model class

    Parameters
    ----------
    struct_connectivity: float 2D array, optional
    	Structural connectivty matrix, a weighted adjacency matrix. The default is the AAL90 matrix
    delays_matrix: float 2D array, optional
    	Matrix of the delays between connections. The default is the AAL90 distances matrix in meters.
    K: float, optional 
    	Global coupling parameter **K** (it will be normalized by the number of nodes). The default is 5.
    dt: float, optional
    	Integration time step. 
        The default is 1e-4 seconds.
    simulation_period: float, optional
    	Total time of the simulation. 
        The default is 100 seconds.
    StimTstart: float, optional. 
    	Starting time of the stimulaiton
        (PFDK only! The default is 0 seconds.)
    StimTend: float, optional. 
    	Final time of the stimulation 
        (PFDK only! The default is 0 seconds.)
    StimFreq: float, optional. 
    	Stimulation frequency, **sigma** 
        (PFDK only! The default is 0 seconds.) 
    StimWeigth: float, optional. 
    	Stimulation force amplitude, **F** 
        PFDK only! The default is 0 seconds.)
    n_nodes: int, optional.
    	Number of oscillatory nodes. 
        The default is 90 nodes.	
    natfreqs: float 1D array, optional.
    	Natural frequencies, **omega_n**, of the system oscillators. 
        The default is None, in order to use the **nat_freq_mean** and **nat_freq_std** to build the array.
    nat_freq_mean: float, optional.
    	Average natural frequency. The default is 0 Hz (only used if **natfreqs** is None).
    nat_freq_std: float, optional.
    	The standard deviation for a Gaussian distribution. The dafult is 2 Hz (only used if natfreqs is None).
    GenerateRandom: boolean, optional.
    	Set if the natural frequencies comes from the same random seed for each simulation.
    SEED: int, optional.
    	The simulation seed. The default is 2. 	 
    mean_delay: float, optional.
    	The mean delay to scale the distance matrix. It also could be seen as the inverse of the conduction speed. 
        The default is 0.1 seconds/meter. 
    	
    Returns
    -------
    model: Kuramoto
    	Kuramoto model that implements itself integration method
    """
    def __init__(self,
                struct_connectivity=None,
                delays_matrix=None,
                K=5,
                dt=1.e-4,
                simulation_period=100,
                StimTstart=0,StimTend=0,StimFreq=0,StimWeigth=0,
                n_nodes=90,natfreqs=None,
                nat_freq_mean=0,nat_freq_std=2,
                GenerateRandom=True,SEED=2,
                mean_delay=0.10):
        """
        struct_connectivity: is the weigthed adjacency matrix (struct_connectivity).
        K: is the global coupling strength.
        simulation_period    : is the simulation time.
        dt: is the integration time step.
        StimFreq: is the stimulation frequency. 
        StimWeigth: is the stimulation Amplitude.
        StimTstart: is the onset start time.
        StimTend  : is the offset time.
        n_nodes: is the number of nodes.
        natfreqs: is the natural frequencies of the nodes. 
        delays_matrix: is the delay matrix.
        mean_delay: is the mean delay (in Cabral, this delay is a scale like global coupling strength)
        random_nat_freq: random natural frequencies every time if True, if False
        SEED: guarantees that the natfreqs are the same at each run
        """
        if n_nodes is None and natfreqs is None:
            raise ValueError("n_nodes or natfreqs must be specified")
        self.n_nodes=n_nodes
        self.mean_delay=mean_delay
        self.dt=dt
        self.simulation_period=simulation_period
        self.K=K
        self.StimTstart=StimTstart
        self.StimTend=StimTend
        self.StimFreq=StimFreq*2*np.pi
        self.StimWeigth=StimWeigth
        self.ForcingNodes=np.zeros((self.n_nodes,1))
        self.nat_freq_mean=nat_freq_mean
        self.nat_freq_std=nat_freq_std
        self.seed=SEED
        self.random_nat_freq=GenerateRandom
        self.noise_std=0.0
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
        """
        Scale the delays matrix by the mean_delay factor.
        If **mean_delay** ==0, the delay matrix becomes a zeros matrix
        in other case, the delays matrix is divided by the its mean valuem
        and then multiplied by the scaling factor.

        
        Returns
        -------
        None.

        """
        
        #
        if self.mean_delay==0:
            self.delays_matrix=np.zeros((self.n_nodes,self.n_nodes))
        else:
            self.delays_matrix=self.delays_matrix/np.mean(self.delays_matrix[self.struct_connectivity>0])*self.mean_delay
    
    def initializeNatFreq(self,natfreqs):
        """
        Set the vector of the natural frequencies of the oscillators with:
            Single value por all nodes 

        Parameters
        ----------
        natfreqs : float (single | 1D array) 
           Natural frequencies of the oscillators, could be a single value if 
           it is the same for all the nodes.

        Returns
        -------
        natfreqs : float 1D array
            N x 1 array with the natural frequencies of the oscillators.

        """
        
        if natfreqs is not None:
            if type(natfreqs)==int or type(natfreqs)==float or type(natfreqs)==np.float32:
                #Equal frequency for all nodes
                natfreqs=natfreqs*np.ones(self.n_nodes) 
            elif self.n_nodes == len(natfreqs):
                #Frequencies specified in an array of the same length than n_nodes
                natfreqs=natfreqs
                print("Assigned intrinsic frequencies for each node")
            elif type(natfreqs)==str:
                if natfreqs.find('mat')!=-1:
                    natfreqs=loadmat(natfreqs)['natfreqs'][:,0]
                else:
                    natfreqs=np.array(eval(natfreqs))
                if self.n_nodes == len(natfreqs):
                    #Frequencies specified in an array of the same length than n_nodes
                    natfreqs=natfreqs
                    print("Assigned intrinsic frequencies for each node")
                else:
                    print('Natural frequencies are bad defined')
            else:
                print('Natural frequencies are bad defined')
        else:
            #Generate random natural frequencies from a Gaussian distribution
            if self.random_nat_freq==True:
                natfreqs=np.random.normal( self.nat_freq_mean,self.nat_freq_std ,size=self.n_nodes)
            else:
                np.random.seed(self.seed)
                natfreqs=np.random.normal( self.nat_freq_mean,self.nat_freq_std ,size=self.n_nodes)
                
        return natfreqs

    def initializeForcingNodes(self,forcingNodes):
        """
        Set the binary vector of forcing/no-forcing nodes
        

        Parameters
        ----------
        forcingNodes : list
            List of the indexes of the forcing nodes.

        Returns
        -------
        None.

        """
        
        self.ForcingNodes=np.zeros((self.n_nodes,1))
        if forcingNodes is not None:
            if type(forcingNodes)==int:
                self.ForcingNodes[forcingNodes]=1
            elif type(forcingNodes)==str:
                fnodes=eval(forcingNodes)
                for node in fnodes:
                    self.ForcingNodes[node]=1
            else:
                for node in forcingNodes:
                    self.ForcingNodes[node]=1    

    def loadParameters(self,parameters):
        """

        Load the parameters of the model
        
        Parameters
        ----------
        parameters : dict
            Dictionary with the parameters.

        Returns
        -------
        None.

        """
        
        self.K=parameters['K'] #global coupling
        self.n_nodes=parameters['n_nodes'] #Number of nodes
        self.mean_delay=parameters['mean_delay'] #mean delay
        self.dt=parameters['dt'] #simulation/storage time step
        self.simulation_period=parameters['simulation_period'] #Duration of stimulus
        self.StimTstart=parameters['StimTstart'] #starting time of stimulation
        self.StimTend=parameters['StimTend'] #ending time of stimulation
        self.StimWeigth=parameters['StimWeight'] #Amplitude of stimulation
        self.StimFreq=parameters['StimFreq'] #Frequency of the stimulation 
        self.StimFreq=2*np.pi*self.StimFreq
        self.seed=parameters['seed'] #random seed
        self.noise_std=parameters['noise_std']#noise
        self.nat_freq_mean=parameters['nat_freq_mean'] #mean of the natural frequencies
        self.nat_freq_std=parameters['nat_freq_std'] #deviation of the natural frequencies
        self.random_nat_freq=parameters['random_nat_freq'] #Flag: random nat. frequency for each realization
        nat_freqs=parameters['nat_freqs'] #natural frequencies
        try:
            self.initializeForcingNodes(parameters['ForcingNodes'])
        except:
            self.initializeForcingNodes(None)
        self.natfreqs=self.initializeNatFreq(nat_freqs)
        if parameters['struct_connectivity']=='AAL90':
            self.struct_connectivity=self.load_struct_connectivity()
            self.delays_matrix=self.initializeTimeDelays()
        else:
            self.struct_connectivity=matrices.loadConnectome(self.n_nodes,parameters['struct_connectivity']) #structural connectivity matrix
            self.delays_matrix=matrices.loadDelays(self.n_nodes,parameters['delay_matrix']) #delay matrix
        self.applyMean_Delay()
        self.ω = 2*np.pi*self.natfreqs #from Hz to rads
        
        self.global_coupling=self.K/self.n_nodes #scaling of the global coupling

    def initializeTimeDelays(self):
        """
        
        Initialize the default matrix **D**: the delays matrix of AAL90.
        Only if the matrix was not specified in the input parameters
    
        Returns
        -------
        D : float 2D array
            Delays matrix.
    
        """
        
        No_nodes=self.n_nodes
        D = loadmat('../input_data/AAL_matrices.mat')['D']
        D=D[-No_nodes:,-No_nodes:]
        if No_nodes>90:
            print('Max No. Nodes is 90')
            No_nodes=90
        D /= 1000 # Distance matrix in meters
        self.delays_matrix=D
        return D
    
    def load_struct_connectivity(self):
        """
        Intialize the default structural connectivity matrix **C** from AAL90
        Only if the matrix was not specified in the input parameters
        
        Returns
        -------
        C : float 2D array
            Structural connectivity matrix.
    
        """
        
        n=self.n_nodes
        
        C = loadmat('../input_data/AAL_matrices.mat')['C']
        if n>90:
            print('Max No. Nodes is 90')
            n=90
        C=C[-n:,-n:]
        C[np.diag(np.ones(n))==0] /= C[np.diag(np.ones(n))==0].mean()
    
        return C
    
    def setMeanTimeDelay(self, mean_delay):
        """
        Set the **mean_delay** parameter
    
        Parameters
        ----------
        mean_delay : float
            Mean delay value that scales the delays matrix **D**.
    
        Returns
        -------
        None.
    
        """
        self.mean_delay=mean_delay
        self.applyMean_Delay()
        
    def setNoiseSD(self,noise):
        self.noise_std=noise
        
    def setGlobalCoupling(self,K):
        """
        Set the global coupling parameter **K**
    
        Parameters
        ----------
        K : float
            Scaling factor for the structural connectivity matrix **C**.
            The value using this function is scaled by the number of nodes
            then **K** =K/N.
    
        Returns
        -------
        None.
    
        """
        
        self.K=K
        self.global_coupling=self.K/self.n_nodes
        
    def vectdiff(self,x,D):
        N=np.shape(x)[0]
        diff=np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                diff[j,i]=x[j,D[i,j]]-x[i,0]
        return diff
    
    def defineA(self):
        self.A=self.global_coupling*self.struct_connectivity
    
    def defineB(self):
        self.B=np.zeros_like(self.struct_connectivity)
        for i in range(self.n_nodes):
            if self.ForcingNodes[i]==1:
                self.B[i,i]=self.StimWeigth
    
    def defineD(self):
        self.D=(np.ceil(self.delays_matrix/self.dt)).astype(int)
        
    def dotx(self,x,u):
        diff_theta=self.vectdiff(x,self.D)
        dotx=self.ω+np.diag(np.matmul(self.A,np.sin(diff_theta)))+np.matmul(self.B,u)[:,0]
        return dotx,diff_theta
    
    def setRank(self,r):
        self.rank=r
    
    def setDesiredPoles(self,poles):
        self.desired_eigs=poles
        
    def simulate(self):
        self.defineA()
        self.defineB()
        self.defineD()
        
        T=int(self.simulation_period//self.dt)
        self.x=np.zeros((self.n_nodes,T))
        
        self.x_dot=np.zeros((self.n_nodes,T))
        self.diff_theta=np.zeros((self.n_nodes,self.n_nodes,T))
        
        #Without control, the external input is zero
        self.u_Out=np.zeros((self.n_nodes,T))
        #Initial conditions
        np.random.seed(self.seed)
        self.x[:,0]=np.random.rand(self.n_nodes)*2*np.pi
        
        for n in tqdm(range(1,T)):
            self.x_dot[:,n-1],self.diff_theta[:,:,n-1]=self.dotx(self.x,self.u_Out[:,n-1:n])
            self.x[:,1::]=self.x[:,0:-1]
            self.x[:,0:1]+=self.dt*self.x_dot[:,n-1:n]
            ##################################################################
            ############# CONTROL ##################
            # Parameters
            L=50 #Window to linearize
            M=3 #Repetitions to get the average
            ########################################
            # At each window (fixed length)
            if n>=self.StimTstart/self.dt and n<=self.StimTend/self.dt:
                if n%L==0:
                    
                    # 1. Linearize
                    try:
                        seigs, PHI_b, eigsA, mean_bigA,mean_bigB, smallA, smallB=dmd.networkDMD(np.sin(self.x[:,0:L]),C=self.struct_connectivity,M=M,u=self.u_Out[:,n-L:n],drive_nodes=np.arange(90),rankX=-1,rankY=self.rank,dt=self.dt,returnMatrices=True)
                    except:
                        seigs, PHI_b, eigsA, mean_bigA,mean_bigB, smallA, smallB=dmd.networkDMD(np.sin(self.x[:,L+1:2*L+1]),C=self.struct_connectivity,M=M,u=self.u_Out[:,n-L:n],drive_nodes=np.arange(90),rankX=-1,rankY=self.rank,dt=self.dt,returnMatrices=True)
                   
                    A,B_,C,D=control.build_tf_NDMD(seigs,PHI_b, mean_bigB, self.dt)
                    B=np.copy(B_)
                    for j in range(self.n_nodes):
                        B[j*self.rank:(j+1)*self.rank,j:j+1]=1
                    B_control=np.ones((self.rank,1))
                    K,PHIA=control.build_K_Ackerman(smallA,B_control,self.desired_eigs,self.dt,self.rank,N=self.n_nodes)
                    # 2. Simulate the LTI
                    t=np.arange(0,(2*L+1)*self.dt,self.dt) #2 windows
                    x_windowed_controlled=np.zeros((self.n_nodes*self.rank,len(t)),dtype=complex)
                    y_windowed_controlled=np.zeros((self.n_nodes,len(t)))
                    p=np.zeros((self.n_nodes,len(t)))
                    p[np.argwhere(self.ForcingNodes==1)[:,0],:]=np.sin(2*np.pi*self.StimFreq*t)
                  
                    u_internal=np.zeros((self.n_nodes,len(t)))
                    K_control=K[0:1,0:self.rank]
                    G=PHI_b@np.linalg.inv(np.eye(self.rank)-smallA+B_control@K_control)
                    K0=1/np.sum(G,axis=1)
                    
                    for nn,tt in enumerate(t[1::]):
                        #Control Ackerman
                        if nn==0:
                            u_internal[:,0]=1
                        else: 
                            K2=1/self.StimWeigth
                            K3=0.8
                            u_internal[:,nn]=np.real((K2*p[:,nn]-K3)-K0*np.eye(self.n_nodes)@K@x_windowed_controlled[:,nn])
                        x_windowed_controlled[:,nn+1]=A@x_windowed_controlled[:,nn]+B@u_internal[:,nn]
                        y_windowed_controlled[:,nn+1]=np.real(C@x_windowed_controlled[:,nn]+D@u_internal[:,nn])
                    # import matplotlib.pyplot as plt
                    # plt.figure()
                    # plt.plot(y_windowed_controlled[7,:])
                    # plt.plot(p[7,:])
                    
                    if (np.shape(self.u_Out)[1]-L)>n:
                        # import matplotlib.pyplot as plt
                        # plt.plot(u_internal[7,:])
                        self.u_Out[:,n:n+L]=u_internal[:,L:2*L] #Test with the arcsin
                    else:
                        Ltail=np.shape(self.u_Out)[1]-n
                        self.u_Out[:,n::]=u_internal[:,L:L+Ltail]
            
            