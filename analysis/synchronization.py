#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from scipy.io import loadmat
from scipy.stats import stats
import scipy.linalg as linalg
import scipy.signal as signal
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


#### Additional functions ##################
def shannonEntropy(p):
    """
    Calculates the Shannon entropy 
    of the probability mass distribution p[j]

    Parameters
    ----------
    p : float 1D array
        Probability mass distirbution. All the elements must sum 1.

    Returns
    -------
    S : float
        Shannon Entropy.

    """ 
    S=0
    for j in range(len(p)):
        if p[j]>0:
            S-=p[j]*np.log(p[j])
    return S

def shannonEntropyTimeMatrix(X,nbins=100,bin_start=None,bin_end=None):
    """
    Calculates the shanon entropy 
    of the probability mass distribution p[j].
    If **bin_start** and **bin_end**, are not specified, this functions actuates similar as np.histogram
    to get the histogram of the *i,j* element of the tensor **X**.

    Parameters
    ----------
    X : float 3D array 
        A tensor of size TxMxN where T is the number of time points
    nbins : int, optional
        Number of bins to discretize the element *i,j* of the tensor X. The default is 100 bins.
    bin_start : float, optional
        Start value of the bins range. The default is None. If None
    bin_end : float, optional
        End value of the bins range. The default is None.        
    
    Returns
    -------
    S : float 2D array
        Shannon Entropy matrix.

    """ 
    if bin_start==None and bin_end==None:
        bins=nbins
    else:
        bins=np.linspace(bin_start,bin_end,nbins)
    M=np.shape(X)[1]
    N=np.shape(X)[2]
    S=np.zeros((M,N))
    for i in range(M):
        for j in range(N):
            p,_=np.histogram(X[:,i,j],bins=bins,density=True)
            if i!=j:
                p=p*(180/nbins)
            else:
                p=p/nbins
            #Checkpoint: all distirbuitions must sum 1
            #print(np.sum(p))
            Sij=0        
            for n in range(len(p)):
                if p[n]>0:
                    Sij-=p[n]*np.log(p[n])
            S[i,j]=Sij
    return S

######Synchrony Measurements############################

#Scalar

def elementFunctionalConnectivity(x1,x2):
    """
    Calculate the element *i,j* of the functional connectivity matrix
    
    Parameters
    ----------
    x1 : float 1D array
        First vector of data with lenghth L.
    x2 : float 1D array
        Second vector of data with lenghth L.

    Returns
    -------
    fc : float
        Pearson coefficient between **x1** and **x2**, from -1 to 1.

    """
    fc,p=stats.pearsonr(x1, x2)
    return fc


def elementFunctionalConnectivityTheta(x1,x2):
    """
    Calculate the *i,j* element of the functional connectivity matrix
    after applying the *sin()* function to the data's vectors.
    
    Parameters
    ----------
    x1 : 1D array: float
        First vector of data with lenghth L.
    x2 : 1D array: float
        Second vector of data with lenghth L.

    Returns
    -------
    fc : float
        Pearson coefficient between **x1** and **x2**, from -1 to 1.

    """
    fc,p=stats.pearsonr(np.cos(x1), np.cos(x2))
    return fc



def entropySynchrony(x1,x2=None,n=1,m=1):
    """
    Index of synchrony based in the Shannon entropy
    If **bin_start** and **bin_end**, are not specified, this functions actuates similar as np.histogram
    to get the histogram of the *i,j* element of the matrix **X**.
    Parameters
    ----------
    x1 : float 1D array
        First vector of data with lenghth L.
    x2 : float 1D array.
        First vector of data with lenghth L.
    n : int, optional
        Harmonic of x1 signal. The default is 1.
    m : int, optional
        Harmonic of x2 signal. The default is 1.

    Returns
    -------
    rho : float
        Synchrony index from 0 t0 1.
    """
    if x2==None:
        phi=x1
    phi=np.abs(n*x1-m*x2)%(2*np.pi)
    L=len(x1)
    N=int(np.exp(0.624+0.4*np.log(L-1)))
    hist,bins=np.histogram(phi,bins=np.linspace(0, 2*np.pi,N),density=True)
    rho=1-shannonEntropy(hist*(bins[1]-bins[0]))/(np.log(N))
    return rho

def entropySynchronyMatrix(X,nbins=100,bin_start=None,bin_end=None):
    """
    Calculate the entropy index of synchrony

    Parameters
    ----------
    X : float 2D array
        matrix of data with size TxN.
    nbins : int, optional
        Number of bins to discretize the element *i,j* of the tensor X. The default is 100 bins.
    bin_start : float, optional
        Start value of the bins range. The default is None. If None
    bin_end : float, optional
        End value of the bins range. The default is None.  

    Returns
    -------
    rho : float
        Entropy synchrony measurement, values from 0 to 1.

    """
    
    if bin_start==None and bin_end==None:
        bins=nbins
    else:
        bins=np.linspace(bin_start,bin_end,nbins)
    L=np.shape(X)[0]
    S=shannonEntropyTimeMatrix(X,nbins=nbins,bin_start=bin_start,bin_end=bin_end)
    N=int(np.exp(0.624+0.4*np.log(L-1)))
    rho=1-S/np.log(N)
    return rho

def fourierModeIndex(x1,x2,n=1,m=1):
    """
    Fourier index based in the trigonometry 
    identity sin^2(alpha)+cos^2 (alpha)=1

    Parameters
    ----------
    x1 : 1D array: float
        First vector of data with lenghth L.
    x2 : 1D array: float
        Second vector of data with lenghth L.
    n : int, optional
        Harmonic of x1 signal. The default is 1.
    m : int, optional
        Harmonic of x2 signal. The default is 1.

    Returns
    -------
    gamma : float.

    """
    
    phi=np.abs(n*x1-m*x2)
    gamma=np.sqrt(np.mean(np.cos(phi))**2+np.mean(np.sin(phi))**2)
    return gamma


#Array
def completeSynchrony(x1,x2):
    """
    Absolute error between two signals.
    From the use of native Python operators, the arrays could be of any dimension while
    *np.shape* ( **x1** )==*np.shape* ( **x2** )

    Parameters
    ----------
    x1 : float 1D (ND) vector
        Array of data.
    x2 : float 1D (ND) vector
        Array of data.

    Returns
    -------
    error : float
        Absolute error.

    """
    return np.abs(x1-x2)
    
    
    
def phaseLockingValueTwoNodes(x1,x2):
    """
    Phase locking between x1 and X2

    Parameters
    ----------
    x1 : float 1D array
        First vector of data with lenghth L.
    x2 : float 1D array
        Second vector of data with lenghth L.

    Returns
    -------
    plv : float
        phase locking value

    """
    T=len(x1)
    es1s2=np.exp(1j*(x1-x2))
    plv=np.abs(np.sum(es1s2))/T
    return plv


def phaseLockingValueMatrix(X):
    """
    Phase locking value for pairs in matrix of phases X 

    Parameters
    ----------
    X : float 2D array
        Array of data of size Nodes x Time 

    Returns
    -------
    plv : float 2D array
        Phase locking value matrix

    """
    N=np.shape(X)[0]
    T=np.shape(X)[1]
    plv=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            plv[i,j]=phaseLockingValueTwoNodes(X[i,:]%(2*np.pi),X[j,:]%(2*np.pi))
    return plv

def phaseLockingDiffPhase(diffX):
    """
    Phase locking value given the phase difference matrix 

    Parameters
    ----------
    diffX : float 3D array
        Time X Nodes X Nodes difference between phases

    Returns
    -------
    plv : float 2D array
        phase locking value matrix
    """
    T=np.shape(diffX)[0]
    plv=np.abs(np.sum(np.exp(1j*diffX),axis=0)/T)
    return plv

def conditionalProbabiliy(x1,x2,theta=1.999*np.pi,epsilon=0.001,n=1,m=1):
    """
    Conditional probability

    Parameters
    ----------
    x1 : 1D array
        float Lenghth L.
    x2 : float 1D array
        Length L.
    theta : float, optional
         Target-phase at where the signals must coincide. 
         The default is 1.999*np.pi.
    epsilon: float, optional
        error around the target-phase
    n : int, optional
        Harmonic of x1 signal. The default is 1.
    m : int, optional
        Harmonic of x2 signal. The default is 1.
    
    Returns
    -------
    eta : float
        Percentage of coincident points

    """
    index_sort=np.argwhere((x1%(2*np.pi*m))>=theta)
    x2_values=x2[index_sort]%(2*np.pi*n)
    M=len(x2_values)
    eta=0
    for m in range(M):
        if np.abs(x2_values[m]-theta)<epsilon:
            eta+=1.0/M    
    return eta



def KuramotoOrderParameter(X,δ=-1,σ=0,T=-1,dt=-1):
    """
    Kuramoto order parameter, 
    optional: subnetwork with indexes δ
    optional: frequency correction σ=stim frequency, T: tmax, dt:Ts 
    
    Parameters
    ----------
    X : 2D array
        Nodes x time.
    δ : int 1D array, optional
        nodes indexes. The default is -1.
    σ : float, optional
        stim frequency. The default is 0.
    T : float, optional
        t_max. The default is -1.
    dt : float, optional
        sampling time. The default is -1.

    Returns
    -------
    KOP : float
        Kuramoto order parameter. Between 0 and 1.

    """
    
    if δ==-1:
        δ=np.arange(0,np.shape(X)[0])
    Xt=X[δ,:]
    σt=0
    if T!=-1 and dt!=-1:
        input_times=np.linspace(0, T,int(T//dt)+1)
        σt=σ*input_times
    Xt+=σt #Sum the same vector to each row
    KOP=np.abs(np.mean(np.exp(1j*Xt),axis=0))
    return KOP

def localOrderParameter(x):
    """
    Local order parameter
    Relation of each node phase with the global order parameter
    
    Parameters
    ----------
    x : 2D array
        Nodes x time.

    Returns
    -------
    LOP: float 1D array
        Array with values of the local order parameter.

    """
        
    r=KuramotoOrderParameter(x)
    abs_r=np.abs(r)
    conj_r=np.conjugate(r)
    zeta=1j*np.exp(1j*x)*conj_r/abs_r
    return np.mean(zeta,axis=1)



def coherence(x,nperseg=4096,noverlap=3600,fs=1000):
    """
    Coherence for each pair of signals in x
    
    Parameters
    ----------
    x : float 2D array.
        Nodes x Time.
    nperseg : int, optional
        Number of samples of the time window. The default is 4096.
    noverlap : int, optional
        Number of samples of the overlap window. The default is 3600.

    Returns
    -------
    freqs: float 1D array
        frequency bins 
    Cxx : float 3D array
        Tensor of coherences of each pair of nodes.
        Node x Node x Frequency

    """
    
    Cxx=np.zeros((np.shape(x)[0],np.shape(x)[0],nperseg//2+1))
    for i in range(np.shape(x)[0]):
        for j in range(np.shape(x)[0]):
            freqsc,Cxx[i,j,:]=signal.coherence(np.cos(x[i,:]),np.cos(x[j,:]),fs=fs,nperseg=nperseg,noverlap=noverlap)
    return freqsc, Cxx



def synchronyTwoNodes(x1,x2,n=1,m=1):
    """
    Return multiple synchrony measurements for two vector of data with the same length

    Parameters
    ----------
    x1 : float 1D array
        First data vector with length L.
    x2 : float 1D array
        Second data vector with length L.
    n : int, optional
        Mode (Harmonic) of **x1**. The default is 1.
    m : int, optional
        Mode (Harmonic) of **x2**. The default is 1.

    Returns
    -------
    plv : float
        Phase locking value.
    gamma : float
        Fourier mode index.
    phi : float
        Absolute error.
    rho : float
        Entropy Synchrony.

    """
    
    #absolute error
    phi=np.abs(n*x1-m*x2)%(2*np.pi)
    #Shanon entropy
    rho=entropySynchrony(x1,x2,n=n,m=m)
    #Fourier Index
    gamma=fourierModeIndex(x1,x2,n=n,m=m)
    #Phase locking value
    plv=phaseLockingValueTwoNodes(x1, x2)
    return plv,gamma, phi, rho

def synchronyMatrices(X,start_time=0,end_time=20000):
    """
    Return multiple synchrony measurements for a data matrix with size NxT
       
    Parameters
    ----------
    X : float 2D array
        Data matrix of size N x T 
    start_time : int, optional
        Initial time point. The default is 0.
    end_time : int, optional
        Final time point. The default is 20000.
       
    Returns
    -------
    plv_matrix : float 2D array
        Phase locking value array.
    gamma_matrix : float 2D array
        Fourier mode index array.
    phi_matrix : float 2D array
        Absolute error array.
    SE_matrix : float 2D array
        Entropy Synchrony array.
       
    """ 
    N=np.shape(X)[0]
    phi_matrix=np.zeros((N,N,end_time-start_time))
    plv_matrix=np.zeros((N,N,end_time-start_time))
    SE_matrix=np.zeros((N,N))
    gamma_matrix=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            plv_matrix[i,j,:],gamma_matrix[i,j],phi_matrix[i,j,:],SE_matrix[i,j]=synchronyTwoNodes(X[i,start_time:end_time], X[j,start_time:end_time])
     
    return plv_matrix,phi_matrix,gamma_matrix,SE_matrix

def hilbertTheta(X,f_low=0.5,f_high=100,fs=1000,type='butterworth',simulated=True):
    """
    Calculate to envelope and phase of a signal in the frequency band specified by [**f_loww** , **f_high** ] Hz.
    Uses the hilbert transform, then the result has more accuraccy as narrower is the frequency band.

    Parameters
    ----------
    X : float 2D array
        Data matrix of size NxT.
    f_low : float, optional
        Low frequency limit of the pass-band filter. The default is 0.5 Hz.
    f_high : float, optional
        High frequency limit of the pass-band filter. The default is 100 Hz.
    fs : int, optional
        Sampling frequency. The default is 1000 samples/second.
    type : str, optional
        Type of the pass-band filter. The default is 'butterworth', the other option is 'chebysev'.
    simulated : boolean, optional
        Defines if the data comes from simulation, then the impulse response time is removed. The default is True.

    Returns
    -------
    amplitudes : 1D array
        Envelope from the Hilbert transform.
    angles : 1D array
        Phase time serie from the Hilbert transform.

    """
    
    #Define the filter 2nd order
    if type=='butterworth':
        b,a=signal.butter(2,[2*f_low/fs,2*f_high/fs],btype='bandpass')
    elif type=='chebysev': 
        b,a=signal.cheby1(2,1e-6,[2*f_low/fs,2*f_high/fs],btype='bandpass')
    #Zero-phase filter
    sinX=np.sin(X)
    #The final order of the filters is 4th
    Xf=signal.filtfilt(b,a,sinX)
    if simulated:
       Xf=Xf[:,fs:-fs]
    #Hilbert transform
    Xfiltered=np.copy(Xf) #Copy of the array as scipy.signal.Hilbert modifies the input 
    Xa=signal.hilbert(Xfiltered,axis=1)
    angles=np.angle(Xa)
    amplitudes=np.abs(Xa)
    return amplitudes, angles


def lowFrequency_envelopes(X,f_low=0.5,f_high=100,fs=1000,simulated=True):
    """
    Returns the envelpes of the Hilbert transform of the signal at a specific frequency

    Parameters
    ----------
    X : float 2D array
        Data matrix of size N x T.
    f_low : float, optional
        Low frequency limit of the pass-band filter. The default is 0.5 Hz.
    f_high : float, optional
        High frequency limit of the pass-band filter. The default is 100 Hz.
    fs : int, optional
        Sampling frequency. The default is 1000 samples/second.
    simulated : boolean, optional
        Defines if the data comes from simulation, then the impulse response time is removed. The default is True.


    Returns
    -------
    envelopes : 2D array
        Low-pass filtered envelopes. Size N x (T-5*fs)

    """
    
    #Signal duration must be higher than 5 seconds, or the same np.shape(X)[1]>5*fs
    #This limitation assures that the filter works well for the frequncy bands where is not signal in simulated data
    #Assume X is NxT
    amplitudes,angles=hilbertTheta(X,f_low=f_low,f_high=f_high,fs=fs,simulated=simulated)
    #The output from HilbertTheta has two seconds less in duration
    #Low-pass 0.5 Hz, removes 1 second of the Analytical signal before filtering
    b,a=signal.butter(4,2*0.5/fs,btype='lowpass')
    envelopes=signal.filtfilt(b,a,amplitudes[:,fs:-fs],axis=1)
    envelopes=envelopes[:,fs//2:-fs//2] #removes half second after filtering
    #Warning! envelopes has three seconds lesser than X
    
    
    return envelopes

def FC_filtered(X,f_low=0.5,f_high=100,fs=1000,simulated=True):
    """
    Calculates the Functional Connectivity matrix from the low-pass filterd envelopes of the 
    signals in X. The envelopes correspond to the frequency band specified by [**f_low** , **f_high** ] Hz.

    Parameters
    ----------
    X : float 2D array
        Data matrix of size NxT.
    f_low : float, optional
        Low frequency limit of the pass-band filter. The default is 0.5 Hz.
    f_high : float, optional
        High frequency limit of the pass-band filter. The default is 100 Hz.
    fs : int, optional
        Sampling frequency. The default is 1000 samples/second.
    type : str, optional
        Type of the pass-band filter. The default is 'butterworth', the other option is 'chebysev'.
    simulated : boolean, optional
        Defines if the data comes from simulation, then the impulse response time is removed. The default is True.


    Returns
    -------
    FC : float 2D array
        Functional connectivity matrix, uses the pearson coefficient, then the values are in the range [-1,1].
    mean_energy : float
        Average of the energy from the envelopes of the signal in **X**.

    """
    #Low frequency envelopes (low-pass at 0.5 Hz of the envelope of
    #the Hilbert transform of the filtered signal between f_low and f_high)
    envelopes=lowFrequency_envelopes(X=X,f_low=f_low,f_high=f_high,fs=fs,simulated=simulated)
    #The mean energy indicates something about how significant is the analysis 
    #in the specific frequency band
    mean_energy=np.mean(envelopes**2)
    N=np.shape(X)[0]
    FC=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i!=j:
                FC[i,j]=elementFunctionalConnectivity(envelopes[i,:],envelopes[j,:])
            else:
                FC[i,j]=1
    return FC, mean_energy

    
    
def FC_filtered_windowed(X,t_start=20000,t_end=40000,f_low=0.5,f_high=100,fs=1000,simulated=True):
    """
    Calculates the Functional Connectivity matrix from the low-pass filterd envelopes of the 
    signals in X. The envelopes correspond to the frequency band specified by [**f_low** , **f_high** ] Hz,
    and between the time points specified by [**t_start** , **t_end** ) samples.

    Parameters
    ----------
    X : float 2D array
        Data matrix of size NxT.
    f_low : float, optional
        Low frequency limit of the pass-band filter. The default is 0.5 Hz.
    f_high : float, optional
        High frequency limit of the pass-band filter. The default is 100 Hz.
    fs : int, optional
        Sampling frequency. The default is 1000 samples/second.
    type : str, optional
        Type of the pass-band filter. The default is 'butterworth', the other option is 'chebysev'.
    simulated : boolean, optional
        Defines if the data comes from simulation, then the impulse response time is removed. The default is True.
    t_start : int, optional
        Initial time point. The default is 20000 samples.
    t_end : int, optional
        Final time point. The default is 40000 samples.
    
    Returns
    -------
    FC : float 2D array
        Functional connectivity matrix, uses the pearson coefficient, then the values are in the range [-1,1].
    mean_energy : float
        Average of the energy from the envelopes of the signal in **X**.

    """
    
    envelopes=lowFrequency_envelopes(X=X,f_low=f_low,f_high=f_high,fs=fs,simulated=simulated)
    #The mean energy indicates something about how significant is the analysis in the specific frequency band
    mean_energy=np.mean(envelopes**2)
    N=np.shape(X)[0]
    FC=np.zeros((N,N))
    time_start=0
    if t_start>2.5*fs:
        time_start=t_start-int(2.5*fs)
    time_end=np.shape(X)[1]
    if t_end-int(2.5*fs)<time_end:
        time_end=t_end-int(2.5*fs)
    for i in range(N):
        for j in range(N):
            if i!=j:
                FC[i,j]=elementFunctionalConnectivity(envelopes[i,time_start:time_end],envelopes[j,time_start:time_end])
            else:
                FC[i,j]=1
    return FC, mean_energy

def diffPhaseHilbert(X,f_low=2,f_high=100,fs=1000,simulated=True):
    """
    Difference of phases from applying the Hilbert Transform in the signals of **X**.

    Parameters
    ----------
    X : float 2D array
        Data matrix of size NxT.
    f_low : float, optional
        Low frequency limit of the pass-band filter. The default is 0.5 Hz.
    f_high : float, optional
        High frequency limit of the pass-band filter. The default is 100 Hz.
    fs : int, optional
        Sampling frequency. The default is 1000 samples/second.
    type : str, optional
        Type of the pass-band filter. The default is 'butterworth', the other option is 'chebysev'.
    simulated : boolean, optional
        Defines if the data comes from simulation, then the impulse response time is removed. The default is True.

    Returns
    -------
    angles float 2D array
        Phase time series.
    diff : float 2D arrat
        Difference of the phases. Size TxNxN

    """
    
    #Assume X is NxT
    amplitudes,angles=hilbertTheta(X,f_low=f_low,f_high=f_high,fs=fs,simulated=simulated)
    N=np.shape(X)[0]
    diff=np.zeros((np.shape(angles)[1],N,N))
    for i in range(N):
        for j in range(N):
            #Absolute Phase difference in degrees (as there is not much sense in define 'forward' signals) 
            diff[:,i,j]=(angles[i,:]-angles[j,:])
    return angles.T, diff   

def diffPhaseTheta(X):
    """
    Difference of phases assuming that **X** contains phase time series.

    Parameters
    ----------
    X : float 2D array
        Matrix of phase time series. Size N x T. 

    Returns
    -------
    angles float 2D array
        The transpose of X and applied a module of 2 pi
    diff : float 2D arrat
        Difference of the phases. Size TxNxN

    """
    
    #Assume X is NxT
    angles=X%(2*np.pi)
    N=np.shape(X)[0]
    diff=np.zeros((np.shape(X)[1],N,N))
    for i in range(N):
        for j in range(N):
            #Absolute Phase difference in degrees (as there is not much sense in define 'forward' signals) 
            diff[:,i,j]=(angles[i,:]-angles[j,:])
    return angles.T, diff   

def absDiffPhase(x):
    """
    The absolute difference between phases time series. The max value in the absolute difference between phases is pi.
    

    Parameters
    ----------
    x : float (ND array)
        Array that contains difference of phases information.

    Returns
    -------
    abs_diff same as x
        Difference of phases in degrees. Useful for the visualization of synchrony with one node as reference.

    """
    
    return np.abs(x)%np.pi*180/np.pi
    

def extract_FCD(data,wwidth=1000,maxNwindows=100,olap=0.9,coldata=False,mode='corr'):
    """
    Created on Wed Apr 27 15:57:38 2016
    @author: jmaidana
    @author: porio
    
    Source: https://github.com/vandal-uv/anarpy/blob/master/src/anarpy/utils/FCDutil/fcd.py
    Functional Connectivity Dynamics from a collection of time series
    
    Parameters
    ----------
    data : array-like
        2-D array of data, with time series in rows (unless coldata is True)
    wwidth : integer
        Length of data windows in which the series will be divided, in samples
    maxNwindows : integer
        Maximum number of windows to be used. wwidth will be increased if necessary
    olap : float between 0 and 1
        Overlap between neighboring data windows, in fraction of window length
    coldata : Boolean
        if True, the time series are arranged in columns and rows represent time
    mode : 'corr' | 'psync' | 'plock' | 'tdcorr'
        Measure to calculate the Functional Connectivity (FC) between nodes.
        'corr' : Pearson correlation. Uses the corrcoef function of numpy.
        'psync' : Pair-wise phase synchrony.
        'plock' : Pair-wise phase locking.
        'tdcorr' : Time-delayed correlation, looks for the maximum value in a cross-correlation of the data series 
        
    Returns
    -------
    FCDmatrix : numpy array
        Correlation matrix between all the windowed FCs.
    CorrVectors : numpy array
        Collection of FCs, linearized. Only the lower triangle values (excluding the diagonal) are returned
    shift : integer
        The distance between windows that was actually used (in samples)
             
    """
    
    if olap>=1:
        raise ValueError("olap must be lower than 1")
    if coldata:
        data=data.T    
    
    all_corr_matrix = []
    lenseries=len(data[0])
    
    Nwindows=min(((lenseries-wwidth*olap)//(wwidth*(1-olap)),maxNwindows))
    shift=int((lenseries-wwidth)//(Nwindows-1))
    if Nwindows==maxNwindows:
        wwidth=int(shift//(1-olap))
    
    indx_start = range(0,(lenseries-wwidth+1),shift)
    indx_stop = range(wwidth,(1+lenseries),shift)
         
    nnodes=len(data)

    for j1,j2 in zip(indx_start,indx_stop):
        aux_s = data[:,j1:j2]
        if mode=='corr':
            corr_mat = np.corrcoef(aux_s) 
        elif mode=='psync':
            corr_mat=np.zeros((nnodes,nnodes))
            for ii in range(nnodes):
                for jj in range(ii):
                    corr_mat[ii,jj]=np.mean(np.abs(np.mean(np.exp(1j*aux_s[[ii,jj],:]),axis=0)))
        elif mode=='plock':
            corr_mat=np.zeros((nnodes,nnodes))
            for ii in range(nnodes):
                for jj in range(ii):
                    corr_mat[ii,jj]=np.abs(np.mean(np.exp(1j*np.diff(aux_s[[ii,jj],:],axis=0))))
        elif mode=='tdcorr':
            corr_mat=np.zeros((nnodes,nnodes))
            for ii in range(nnodes):
                for jj in range(ii):
                    maxCorr=np.max(np.correlate(aux_s[ii,:],aux_s[jj,:],mode='full')[wwidth//2:wwidth+wwidth//2])
                    corr_mat[ii,jj]=maxCorr/np.sqrt(np.dot(aux_s[ii,:],aux_s[ii,:])*np.dot(aux_s[jj,:],aux_s[jj,:]))
        all_corr_matrix.append(corr_mat)
        
    corr_vectors=np.array([allPm[np.tril_indices(nnodes,k=-1)] for allPm in all_corr_matrix])
    
    CV_centered=corr_vectors - np.mean(corr_vectors,-1)[:,None]
    
    
    return np.corrcoef(CV_centered),corr_vectors,shift
    
