#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from scipy.io import loadmat
from scipy.stats import stats
import scipy.linalg as linalg
import scipy.signal as signal
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba import njit

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
        for j in range(i+1,N):
            plv[i,j]=phaseLockingValueTwoNodes(X[i,:]%(2*np.pi),X[j,:]%(2*np.pi))
            plv[j,i]=plv[i,j]
        plv[i,i]=1
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



def cross_spectrum(x,nperseg=4096,noverlap=2048,fs=1000,applySin=False):
    """
    Real part of the cross-spectrum for each pair of signals in x
    
    Parameters
    ----------
    x : float 2D array.
        Nodes x Time.
    nperseg : int, optional
        Number of samples of the time window. The default is 4096.
    noverlap : int, optional
        Number of samples of the overlap window. The default is 3600.
        With None, the nperseg is used as value of nfft of a FFT with zero-padding
    fs: int, optional
        Sampling frequency
    applySin: boolean, optional
        Apply the sin function before performing the cross-spectrum calculation
    Returns
    -------
    freqs: float 1D array
        frequency bins 
    Cxx : float 3D array
        Tensor of coherences of each pair of nodes.
        Node x Node x Frequency

    """
    if applySin:
        x=np.sin(x)
    nfft=nperseg
    if noverlap==None:
        nperseg=None
    Cxx=np.zeros((np.shape(x)[0],np.shape(x)[0],nfft//2+1))
    for i in range(np.shape(x)[0]):
        for j in range(i+1,np.shape(x)[0]):
            freqsc,Cxx[i,j,:]=np.real(signal.csd(x[i,:],x[j,:],fs=fs,nfft=nfft,nperseg=nperseg,noverlap=noverlap))*(2*fs)/nfft
            Cxx[j,i,:]=Cxx[i,j,:]
        Cxx[i,i,:]=1
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
        for j in range(i+1,N):
            plv_matrix[i,j,:],gamma_matrix[i,j],phi_matrix[i,j,:],SE_matrix[i,j]=synchronyTwoNodes(X[i,start_time:end_time], X[j,start_time:end_time])
            plv_matrix[j,i,:],gamma_matrix[j,i],phi_matrix[j,i,:],SE_matrix[j,i]=plv_matrix[i,j,:],gamma_matrix[i,j],phi_matrix[i,j,:],SE_matrix[i,j]
        plv_matrix[i,i,:]=np.ones((end_time-start_time,))
        gamma_matrix[i,i]=1
        phi_matrix[i,i,:]=np.ones((end_time-start_time,))
        SE_matrix[i,i]=1
    return plv_matrix,phi_matrix,gamma_matrix,SE_matrix

def hilbertFrequencyBand(X,f_low=0.5,f_high=100,fs=1000,type='butterworth',applyTrim=True,applySin=True):
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
    applyTrim : boolean, optional
        Defines if the data comes from simulation, then the impulse response time is removed. The default is True.
    applySin : boolean, optional
        Defines if the sin function must be applied to the data before any processing.
    Returns
    -------
    amplitudes : 1D array
        Envelope from the Hilbert transform. The length is 4*fs less than the length of X.
    angles : 1D array
        Phase time serie from the Hilbert transform. The length is 4*fs less than the length of X.

    """
    
    #Define the filter 2nd order
    if type=='butterworth':
        b,a=signal.butter(2,[2*f_low/fs,2*f_high/fs],btype='bandpass')
    elif type=='chebysev': 
        b,a=signal.cheby1(2,1e-6,[2*f_low/fs,2*f_high/fs],btype='bandpass')
    #Apply the sin function
    if applySin:
        sinX=np.sin(X)
    else:
        sinX=X
    #The final order of the filters is 4th
    Xf=signal.filtfilt(b,a,sinX)
    if applyTrim:
       Xf=Xf[:,2*fs:-2*fs]
    #Hilbert transform
    Xfiltered=np.copy(Xf) #Copy of the array as scipy.signal.Hilbert modifies the input 
    Xa=signal.hilbert(Xfiltered,axis=1)
    angles=np.angle(Xa)
    amplitudes=np.abs(Xa)
    return amplitudes, angles


def envelopesFrequencyBand(X,f_low=0.5,f_high=100,fs=1000,applyTrim=True,applySin=True,applyLow=True,f_lowpass=0.5):
    """
    Returns the envelpes of the Hilbert transform of the signal at a specific frequency
    
    Warning! envelopes has eight seconds lesser than X
    
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
    applyTrim : boolean, optional
        Defines if the data comes from simulation, then the impulse response time is removed. The default is True.
    applyLow: boolean, optional
        Defines if the envelopes are low-pass filtered or if they are directly returned from the Hilbert transform.
    f_lowpass: float, optional
        Defines the stop band of the low-pass filter applied to the Hilbert envelopes
    applySin : boolean, optional
        Defines if the sin function must be applied to the data before any processing.    
    Returns 
    -------
    envelopes : 2D array
        Low-pass filtered envelopes. Size N x (T-5*fs)

    """
    
    #Signal duration must be higher than 5 seconds, or the same np.shape(X)[1]>5*fs
    #This limitation assures that the filter works well for the frequncy bands where is not signal in simulated data
    #Assume X is NxT
    amplitudes,angles=hilbertFrequencyBand(X,f_low=f_low,f_high=f_high,fs=fs,applyTrim=applyTrim, applySin=applySin)
    #The output from hilbertFrequencyBand has two seconds less in duration
    if applyLow:
        #Low-pass 0.5 Hz, removes 1 second of the Analytical signal before filtering
        b,a=signal.butter(2,2*f_lowpass/fs,btype='lowpass')
        envelopes=signal.filtfilt(b,a,amplitudes[:,:],axis=1)
        envelopes=envelopes[:,2*fs:-2*fs] #removes half second after filtering
    else:
        envelopes=amplitudes[:,2*fs:-2*fs] #removes half second after filtering
    
    #Warning! envelopes has eight seconds lesser than X
    
    return envelopes

def FC_filtered(X,f_low=0.5,f_high=100,fs=1000,applyTrim=True,applySin=True):
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
    applyTrim : boolean, optional
        Defines if the data comes from simulation, then the impulse response time is removed. The default is True.
    applySin : boolean, optional
        Defines if the sin function must be applied to the data before any processing.

    Returns
    -------
    FC : float 2D array
        Functional connectivity matrix, uses the pearson coefficient, then the values are in the range [-1,1].
    mean_energy : float
        Average of the energy from the envelopes of the signal in **X**.

    """
    #Low frequency envelopes (low-pass at 0.5 Hz of the envelope of
    #the Hilbert transform of the filtered signal between f_low and f_high)
    envelopes=envelopesFrequencyBand(X=X,f_low=f_low,f_high=f_high,fs=fs,applyTrim=applyTrim,applySin=applySin)
    #The mean energy indicates something about how significant is the analysis 
    #in the specific frequency band
    mean_energy=np.mean(envelopes**2)
    N=np.shape(X)[0]
    FC=np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            FC[i,j]=elementFunctionalConnectivity(envelopes[i,:],envelopes[j,:])
            FC[j,i]=FC[i,j]
        FC[i,i]=1
    return FC, mean_energy

    
    
def FC_filtered_windowed(X,t_start=20000,t_end=40000,f_low=0.5,f_high=100,fs=1000,applyTrim=True,applySin=True):
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
    applyTrim : boolean, optional
        Defines if the data comes from simulation, then the impulse response time is removed. The default is True.
    applySin : boolean, optional
        Defines if the sin function must be applied to the data before any processing.
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
    
    envelopes=envelopesFrequencyBand(X=X,f_low=f_low,f_high=f_high,fs=fs,applyTrim=applyTrim,applySin=applySin)
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
        for j in range(i+1,N):
            FC[i,j]=elementFunctionalConnectivity(envelopes[i,time_start:time_end],envelopes[j,time_start:time_end])
            FC[j,i]=FC[i,j]
        FC[i,i]=1
    return FC, mean_energy

def diffPhaseHilbert(X,f_low=2,f_high=100,fs=1000,applyTrim=True,applySin=False):
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
    applyTrim : boolean, optional
        Defines if the data comes from simulation, then the impulse response time is removed. The default is True.
    applySin : boolean, optional
        Defines if the sin function must be applied to the data before any processing.
    Returns
    -------
    angles float 2D array
        Phase time series.
    diff : float 2D arrat
        Difference of the phases. Size TxNxN

    """
    
    #Assume X is NxT
    amplitudes,angles=hilbertFrequencyBand(X,f_low=f_low,f_high=f_high,fs=fs,applyTrim=applyTrim,applySin=applySin)
    N=np.shape(X)[0]
    diff=np.zeros((np.shape(angles)[1],N,N))
    for i in range(N):
        for j in range(i+1,N):
            #Absolute Phase difference in degrees (as there is not much sense in define 'forward' signals) 
            diff[:,i,j]=(angles[i,:]-angles[j,:])
            diff[:,j,i]=diff[:,i,j]
        diff[:,i,i]=np.zeros((np.shape(angles)[1],))
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

 
def complex_coherence(data_x,data_y,nfft=5000,freq_index=1,wcoh=1000):
    """
    Squared complex coherence, from here is easy to obtain the absolute, the real or the imaginary value

    Parameters
    ----------
    data_x : float array
        data from channel x
    data_y : float array
        data from channel y
    nfft : int, optional
        number of points of the FFT. Defines the resolution of the spectrums. The default is 5000.
    freq_index : int or array int, optional
        indices of the frequencies of interest, relative to nfft. The default is 1.
    wcoh : TYPE, optional
        time window. The default is 1000.

    Returns
    -------
    coh : complex
        Complex average of the coherence in the frequency of interest.

    """
    
    f, Pxx = signal.welch(data_x, nperseg=wcoh, noverlap=wcoh//2+1, nfft=nfft)
    f, Pyy = signal.welch(data_y, nperseg=wcoh, noverlap=wcoh//2+1, nfft=nfft)
    f, Pxy = signal.csd(data_x,data_y,nperseg=wcoh, noverlap=wcoh//2+1, nfft=nfft)
    coh=Pxy/np.sqrt(Pxx*Pyy)
    coh=np.mean(coh[freq_index])
    return coh


def complex_coherence_matrix(data,nfft=5000,freq_index=1,wcoh=1000):
    """
    Squared complex coherence, from here is easy to obtain the absolute, the real or the imaginary value

    Parameters
    ----------
    data_x : 2D float array
        data with shape NxT
    nfft : int, optional
        number of points of the FFT. Defines the resolution of the spectrums. The default is 5000.
    freq_index : int or array int, optional
        indices of the frequencies of interest, relative to nfft. The default is 1.
    wcoh : TYPE, optional
        time window. The default is 1000.

    Returns
    -------
    coh : complex
        Complex average of the coherence in the frequency of interest.

    """
    N=np.shape(data)[0]
    Pxx=np.zeros((N,nfft//2+1))
    coh=np.zeros((N,N),dtype=complex)
    for ii in range(N):
        f, Pxx[ii,:] = signal.welch(data[ii,:], nperseg=wcoh, noverlap=wcoh//2+1, nfft=nfft) 
        for jj in range(ii):
            f, Pxy = signal.csd(data[ii,:],data[jj,:],nperseg=wcoh, noverlap=wcoh//2+1, nfft=nfft)
            coh_f=Pxy/np.sqrt(Pxx[ii,:]*Pxx[jj,:])
            coh[ii,jj]=np.mean(coh_f[freq_index])
    return coh

def abs_coherence_matrix(data,nfft=5000,freq_index=1,wcoh=1000):
    """
    Squared complex coherence, from here is easy to obtain the absolute, the real or the imaginary value

    Parameters
    ----------
    data_x : 2D float array
        data with shape NxT
    nfft : int, optional
        number of points of the FFT. Defines the resolution of the spectrums. The default is 5000.
    freq_index : int or array int, optional
        indices of the frequencies of interest, relative to nfft. The default is 1.
    wcoh : TYPE, optional
        time window. The default is 1000.

    Returns
    -------
    coh : complex
        Complex average of the coherence in the frequency of interest.

    """
    N=np.shape(data)[0]
    Pxx=np.zeros((N,nfft//2+1))
    coh=np.zeros((N,N),dtype=complex)
    for ii in range(N):
        f, Pxx[ii,:] = signal.welch(data[ii,:], nperseg=wcoh, noverlap=wcoh//2+1, nfft=nfft) 
        for jj in range(ii):
            f, coh_f = signal.coherence(data[ii,:],data[jj,:],nperseg=wcoh, noverlap=wcoh//2+1, nfft=nfft)
            coh[ii,jj]=np.mean(coh_f[freq_index])
    return coh

def extract_FCD(data,wwidth=1000,maxNwindows=100,olap=0.9,nfft=5000,freq_index=1,wcoh=100,coldata=False,mode='corr'):
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
    nfft: int
        Number of points of the FFT
    freq_index: int or array int
        frequencies of interest
    wcoh:
        internal window for the coherence
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
    halfnfft=nfft//2+1
    halfwwidth=wwidth//2+1
    if olap>=1:
        raise ValueError("olap must be lower than 1")
    if coldata:
        data=data.T    
    
    
    lenseries=len(data[0])
    
    Nwindows=min(((lenseries-wwidth*olap)//(wwidth*(1-olap)),maxNwindows))
    shift=int((lenseries-wwidth)//(Nwindows-1))
    if Nwindows==maxNwindows:
        wwidth=int(shift//(1-olap))
    
    indx_start = range(0,(lenseries-wwidth+1),shift)
    indx_stop = range(wwidth,(1+lenseries),shift)
         
    nnodes=len(data)
    if mode=='ccoh':
        corr_vectors = np.zeros((len(indx_start),len(np.tril_indices(nnodes,k=-1)[0])),dtype=complex)
    else:
        corr_vectors = np.zeros((len(indx_start),len(np.tril_indices(nnodes,k=-1)[0])))
    
    if wcoh>wwidth:
        print('wcoh must be lower than wwidth')
        return -1,-1,-1
    
    for nmat,(j1,j2) in enumerate(zip(indx_start,indx_stop)):
        aux_s = data[:,j1:j2]
        if mode=='corr':
            corr_mat = np.corrcoef(aux_s)
            corr_mat[np.isnan(corr_mat)]=1
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
        elif mode=='coh':
            coh=complex_coherence_matrix(aux_s,wcoh=wcoh,nfft=nfft,freq_index=freq_index)
            corr_mat=np.abs(coh)
        elif mode=='ccoh':
            coh=complex_coherence_matrix(aux_s,wcoh=wcoh,nfft=nfft,freq_index=freq_index)
            corr_mat=coh
        elif mode=='icoh':
            coh=complex_coherence_matrix(aux_s,wcoh=wcoh,nfft=nfft,freq_index=freq_index)
            corr_mat=np.imag(coh)
        elif mode=='pcoh':
            corr_mat=np.zeros((nnodes,nnodes))
            fourier=np.fft.fft(aux_s,nfft)
            for ii in range(nnodes):
                for jj in range(ii):
                    if len(freq_index)==1:
                        corr_mat[ii,jj]=np.abs(np.mean(np.exp(1j*np.diff(np.angle(fourier[[ii,jj],:][:,freq_index-1:freq_index+1]),axis=0))))
                    else:
                        corr_mat[ii,jj]=np.abs(np.mean(np.exp(1j*np.diff(np.angle(fourier[[ii,jj],:][:,freq_index]),axis=0))))
        elif mode=='tdcorr':
            corr_mat=np.zeros((nnodes,nnodes))
            for ii in range(nnodes):
                for jj in range(ii):
                    maxCorr=np.max(np.correlate(aux_s[ii,:],aux_s[jj,:],mode='full')[wwidth//2:wwidth+wwidth//2])
                    corr_mat[ii,jj]=maxCorr/np.sqrt(np.dot(aux_s[ii,:],aux_s[ii,:])*np.dot(aux_s[jj,:],aux_s[jj,:]))
        corr_vectors[nmat,:]=corr_mat[np.tril_indices(nnodes,k=-1)]
        
    CV_centered=corr_vectors - np.mean(corr_vectors,-1)[:,None]
    FCD=np.corrcoef(CV_centered)
    FCD[np.isnan(FCD)]=1    
    return FCD,corr_vectors,shift
    
def FCD_from_envelopes(X,f_low=8,f_high=13,fs=1000,wwidth=1000,olap=0.9,mode='corr',applyTrim=True,applySin=True):
    """
    Extract the FCD from the low-pass band filtered envelopes 

    Parameters
    ----------
    X : float 2D array
        Data matrix of size NxT.
    f_low : float, optional
        Low frequency limit of the pass-band filter. The default is 5 Hz.
    f_high : float, optional
        High frequency limit of the pass-band filter. The default is 13 Hz.
    fs : int, optional
        Sampling frequency. The default is 1000 samples/second.
    wwidth : integer, optional
        Length of data windows in which the series will be divided to calculate each FC, in samples the default is 1000
    olap : float between 0 and 1, optional
        Overlap between neighboring data windows, in fraction of window length
    mode : str, 'corr' | 'psync' | 'plock' | 'tdcorr'
        Measure to calculate the Functional Connectivity (FC) between nodes.
        'corr' : Pearson correlation. Uses the corrcoef function of numpy.
        'psync' : Pair-wise phase synchrony.
        'plock' : Pair-wise phase locking.
        'tdcorr' : Time-delayed correlation, looks for the maximum value in a cross-correlation of the data series
    applyTrim : boolean, optional
        Defines if the data comes from simulation, then the impulse response time is removed. The default is True.
    
    applySin : boolean, optional
        Defines if the sin function must be applied to the data before any processing.
        
    Returns
    -------
    FCDmatrix : numpy array
        Correlation matrix between all the windowed FCs.
    CorrVectors : numpy array
        Collection of FCs, linearized. Only the lower triangle values (excluding the diagonal) are returned
    shift : integer
        The distance between windows that was actually used (in samples)
    """
    
    envelopes = envelopesFrequencyBand(X=X,f_low=f_low,f_high=f_high,fs=fs,applyTrim=applyTrim,applySin=applySin,applyLow=True)
    
    FCDmatrix, corr_vectors, shift = extract_FCD(envelopes[:,2*fs:-2*fs],wwidth=wwidth,maxNwindows=1000,olap=olap,mode=mode)
    return FCDmatrix, corr_vectors, shift


def extractEventsTimes(x,high_value=1,min_duration=3):
    """
    Returns a list with the starting and ending time points from the  binary time serie 
    of the thresholded events
    
    Parameters
    ----------
    x : 1D int
        Thresholded events or excerpts of the signal at a specific frequency band.
    high_value : int, optional
        The integer that codifies the event. The default is 1 for binary signals.
    min_duration : int, optional
        The minimum duration of the event. The default is 3 samples.

    Returns
    -------
    2D int array
        Array of the events with the index of the node (row 0), starting time (row 1) and ending time (row 2). The size is 3 x N_events.
    """ 
    x[:,-1]=0
    y=np.roll(x,shift=1)
    start_times=np.where((x ==high_value ) & (y < high_value))
    end_times=np.where((x <high_value ) & (y == high_value))
    start_times=np.array(start_times)
    end_times=np.array(end_times)
    end_times[1,:]-=1
    durations=end_times[1,:]-start_times[1,:]
    start_times=start_times[:,durations>=min_duration]
    end_times=end_times[:,durations>=min_duration]
    events_times=np.vstack((start_times,end_times[1,:]))
    return events_times


def high_order_cooccurrences(events_times):
    """
    Find the co-occurrences

    Parameters
    ----------
    events_times : 2D int array
        Array of the events with the index of the node (row 0), starting time (row 1) and ending time (row 2). The size is 3 x N_events.

    Returns
    -------
    co_occurrences : list
        List of the co-occurrent events. 
        Each co-occurrence is a tuple with the starting time point, the duration, and a list with the nodes' indexes. 

    """    
    co_occurrences=[]
    last_end_event=0
    list_events=list(events_times[1,:])
    list_events_without_repetitions=list(dict.fromkeys(list_events))
    for event_start in list_events_without_repetitions:
        if list_events.count(event_start)>1:
            nodes_with_same_start=np.argwhere(events_times[1,:]==event_start)[:,0]
            end_times=events_times[2,nodes_with_same_start]
            last_end_event=np.max(end_times)
            
            node_indexes=events_times[0,nodes_with_same_start]
            nodes_with_different_start=np.argwhere((events_times[1,:]>event_start) & (events_times[2,:]<last_end_event))[:,0]
            node_indexes=np.append(node_indexes,events_times[0,nodes_with_different_start])
            co_occurrences.append([event_start,last_end_event-event_start,[node_indexes]])
    return co_occurrences


def extractTimeStatisticsEvents(X,min_duration=5):
    """
    Extract the co-occurrrences of events in several nodes
    And their time characteristics: fractional occupancy in each node and durations.
    
    Parameters
    ----------
    X : 2D int array
        binary time series of size N(nodes) x T(sampling points). 
        1: indicate the presence of an event. 
        0: ausence
    min_duration : float, optional
        Minimum overlap between the node's events to be considered a co-occurrence. The default is 5 sample points.

    Returns
    -------
    durations : float 
        Duration of each event.
    occupancy : float
        Fractional occupancy of the events in each node N.
    co_occurrences : list
        Each co-occurrence is a tuple with the starting time point, the duration, and a list with the nodes' indexes. 

    """
    N=np.shape(X)[0]
    T=np.shape(X)[1]
    occupancy=np.zeros((N,))
    events_times=extractEventsTimes(X,min_duration=min_duration)
    durations=events_times[2,:]-events_times[1,:]
    occupancy=np.zeros((90,))
    for n in range(N):
        occupancy[n]=np.sum(durations[np.argwhere(events_times[0,:]==n)[:,0]])/T
    co_occurrences=high_order_cooccurrences(events_times)  
    
    return durations, occupancy, co_occurrences

def durationfromLabels(labels,time_window=118,overlap=0.5):
    """
    Calculate the duration of the events from a list of the labels assigned to each time window.
    The method considers that the time windows could have overlap between them.

    Parameters
    ----------
    labels : int array
        Array with the labels from 0 to N-1 of N groups, or clusters.
    time_window : int, optional
        The length of the time window in time units. The default is 118 ms.
    overlap : float, optional
        Overlap between the time windows. The default is 0.5 for 50% overlapping.

    Returns
    -------
    duration_clusters
        A dictionary which keys are the labels, and the items are the duration of the events of each label.

    """
    
    def array2bits(array):
        binary = ''.join(['1' if bit else '0' for bit in array])
        return binary
    def unoverlap_time(x,time_window,overlap):
        return time_window*((1-overlap)*x+overlap)
        
    Nlabels=np.max(labels)+1
    duration_clusters={}
    for label in range(Nlabels):
        duration_clusters[label]=[]
        binarized=labels==label
        events=array2bits(binarized).strip().split('0')
        for event in events:
            if event != '':
                duration = unoverlap_time(len(event),time_window,overlap)
                duration_clusters[label].append(duration)
    return duration_clusters


def transitionsfromLabels(labels):
    """
    Calculate the transitions between events from a list of the labels assigned to each time window.

    Parameters
    ----------
    labels : int array
        Array with the labels from 0 to N-1 of N groups, or clusters.

    Returns
    -------
    transition_matrix : 2d int array
        Matrix with the counts of the transtions.
        The rows indicate the destine
        The columns indicate the orgin

    """
    
    Nlabels=np.max(labels)+1
    transition_matrix=np.zeros((Nlabels,Nlabels))
    for n in range(1,len(labels)):
        if labels[n]!=labels[n-1]:
            tag_destination=labels[n]
            tag_origin=labels[n-1]
            transition_matrix[tag_destination,tag_origin]+=1
    return transition_matrix
    
