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
import time as tm


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
        self.param_sym_func=symengine.Function('param_sym_func')
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
                natfreqs=np.random.normal( self.nat_freq_mean,self.nat_freq_std ,size=self.n_nodes)
            else:
                np.random.seed(self.seed)
                natfreqs=np.random.normal( self.nat_freq_mean,self.nat_freq_std ,size=self.n_nodes)
                
        return natfreqs

    
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
        self.StimWeigth=parameters['StimWeigth'] #Amplitude of stimulation
        self.StimFreq=parameters['StimFreq'] #Frequency of the stimulation 
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
        C=self.struct_connectivity
        mean_delay=self.mean_delay
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

    def param(self,y,arg):
        """
        Auxiliar function in order to present the stimulation from **StimTstart** to **StimTend** 
        Add the maximum of the delays matrix, because for the stored data t_0=max(**delays_matrix**).

        Parameters
        ----------
        y : Not needed, internal function of Jitcdde
            
        arg : Not needed, internal function of Jitcdde 
            
        Returns
        -------
        int(boolean)
            True if the simulation time is inside the stimulation window. 

        """
        

        if self.StimTend>self.simulation_period:
            print("Tend Cannot be larger than the simulation time which is"+'%.1f' %self.T)
        if arg<self.StimTstart+np.max(self.delays_matrix):
            return 0
        elif arg>self.StimTstart+np.max(self.delays_matrix) and arg<self.StimTend+np.max(self.delays_matrix): 
            return 1
        else:
            return 0
            
    def initial_phases(self):
        """
        Set the initial values of the variables

        Returns
        -------
        Initial_values: float 1D array
            Array with the initial values of the ocillators' phases.

        """
        
        return 2*np.pi*np.random.random(size=self.n_nodes)
    
    def kuramotosZero(self,y,t):
        """
        Easiest way to define in python the classical Kuramoto model
        It works (and solves faster) with scipy ODE solvers 

        Parameters
        ----------
        y : float 1D array
            Current (previous) value of the oscillators phases.
        t : float
            Current time (actually not used, but usefull to plot or further analysis).

        Returns
        -------
        dphi_dt : float 1D array
            derivate value at the current time step.

        """
        
        ϴ_i,ϴ_j=np.meshgrid(y,y)
        dphi_dt= self.ω + (self.global_coupling)*( self.struct_connectivity * np.sin( ϴ_j - ϴ_i ) ).sum(axis=0)
        return dphi_dt

    def kuramotos(self):
        """
        Delayed Kuramoto model (from Jitcdde documentation)

        Yields
        ------
        Jitcdde Model
            Delayed Kuramoto. Structural connectivity and delays matrices are required.

        """
        
        for i in range(self.n_nodes):
            yield self.ω[i] +self.global_coupling*sum(
                self.struct_connectivity[i, j] * sin( y(j,t - (self.delays_matrix[i, j])) - y(i))
                for j in range(self.n_nodes) if self.struct_connectivity[i,j]
            )

    def kuramotosForced(self):
        """
        Forced and delayed Kuramoto model

        Yields
        ------
        Jitcdde Model
            Forced and Delayed Kuramoto model. The stimulation parameters are required.

        """
        
        Delta=self.ForcingNodes
        for i in range(self.n_nodes):
            epsilon=np.random.rand(1)[0]
            yield self.ω[i] + self.noise_std*epsilon+Delta[i,0]*self.param_sym_func(t)*self.StimWeigth*sin(self.StimFreq*t-y(i))+self.global_coupling*sum(
                self.struct_connectivity[i, j] * sin( y(j , t - (self.delays_matrix[i, j])) - y(i))
                for j in range(self.n_nodes) if self.struct_connectivity[i,j]
            )


    def IntegrateDD(self):
        """
        Jitcdde solver (integration) function.

        Returns
        -------
        output : float 2D array
            
            Oscillator phases at the sampling times.
            T x N array.   

        """
        
        print('Simulating %d nodes network using K=%.3f and MD=%.3f at f=%.3fHz'%(self.n_nodes,self.K,self.mean_delay,self.nat_freq_mean))
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
        
    ## Solve the classical system with odeint
        #else:
        #    ϴ_o=self.initial_phases()

        #    t = np.linspace(0, self.simulation_period, int(self.simulation_period/self.dt))
        #    dynamics = odeint(self.kuramotosZero, ϴ_o, t)
            
        #    return dynamics


        
        
    def gui_interact(self,K):
        """
        Jupyter notebook function for "online" change of the **K** parameter
        (Note that this method can be used to change any model parameter with the appropiate modifications)

        Parameters
        ----------
        K : float
            Global coupling parameter (it will be normalized by the number of nodes).

        Returns
        -------
        None.

        """
        
        import matplotlib.pyplot as plt
        self.setGlobalCoupling(K) #Modify this line to change another parameter or change more parameters  
        R,Dynamics=self.simulate()
        t=np.linspace(0,self.simulation_period,int(self.simulation_period//self.dt)+1)
        plt.figure(figsize=(12,4))
        plt.plot(t,R)
        plt.show()

    
    def simulate(self,Forced=False):
        """
        Main function. Simulate the Kuramoto model.
        To avoid the additional overload from using the Partially Forced Kuramoto you shoul pass 
        False in the Forced argument.
        On the other hand, if you need to apply stimulation, use Forced=True 

        Parameters
        ----------
        Forced : boolean, optional
            Set if the simulation is going to use or not the Partially Forced Kuramoto. The default is False.

        Returns
        -------
        R : float 1D array
            Kuramoto order parameter. Array with dimensions T x 1.
        dynamics : float 2D array
            Kuramoto oscillators phases. Array with dimensions T x N

        """
        
        self.Forced=Forced
        dynamics=self.IntegrateDD()
        R=self.calculateOrderParameter(dynamics)
        return R,dynamics

    def testIntegrateForced(self):
        """
        Test the Partially forced and delayed model

        Returns
        -------
        None.

        """
        
        R,_=self.simulate(Forced=True)
        print(np.mean(R))
        
    def testIntegrate(self):
        """
        Test the Delayed Kuramoto

        Returns
        -------
        None.
        
        """
        
        R,_=self.simulate(Forced=True)
        print(np.mean(R))

    
    @staticmethod
    def phase_coherence(angles_vec):
        """
        Calculate the Kuramoto Order parameter for each time step

        Parameters
        ----------
        angles_vec : float 1D array
            Phases of the oscillators at time t.

        Returns
        -------
        R(t): float            
        	A time point of the Kuramoto order parameter: abs(exp{(theta_n(t))}) 

        """
        
        suma=sum([np.exp(1j*i) for i in angles_vec ])
        return abs(suma/len(angles_vec))


    def calculateOrderParameter(self,dynamics):
        """
        Calculate Kuramoto order parameter

        R=abs((1/N)* \\sum_{n} ( e^{(i*\\theta_{n})}))

        Parameters
        ----------
        dynamics : float 2D array
            T x N matrix with the oscillators phases.

        Returns
        -------
        R : float 1D array
            Kuramoto order parameter.
        """
        R=[self.phase_coherence(vec) for vec in dynamics]
        del dynamics
        gc.collect()
        return R
