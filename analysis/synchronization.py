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
    Calculates the shanon entropy 
    #of the probability mass distribution p[j]

    Parameters
    ----------
    p : 1D array: float
        probability mass distirbution. All the elements sum 1.

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
    #of the probability mass distribution p[j]

    Parameters
    ----------
    X : 3D array: float 
        size TxMxN where T is the number of time points

    Returns
    -------
    S : 2D array: float
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
    Element of the functional connectivity matrix
    Parameters
    ----------
    x1 : 1D array: float
        Lenghth L.
    x2 : 1D array: float
        Length L.

    Returns
    -------
    fc : float
        Pearson coefficient between x1 and x2, from -1 to 1.

    """
    fc,p=stats.pearsonr(x1, x2)
    return fc


def elementFunctionalConnectivityTheta(x1,x2):
    """
    Element of the functional connectivity matrix
    Parameters
    ----------
    x1 : 1D array: float
        Lenghth L.
    x2 : 1D array: float
        Length L.

    Returns
    -------
    fc : float
        Pearson coefficient between x1 and x2, from -1 to 1.

    """
    fc,p=stats.pearsonr(np.cos(x1), np.cos(x2))
    return fc



def entropySynchrony(x1,x2=None,n=1,m=1):
    """
    Index of synchrony based in the Shannon entropy
    ----------
    x1 : 1D array: float
        Lenghth L.
    x2 : 1D array: float
        Length L.
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
    Calculates the shanon entropy 
    #of the probability mass distribution p[j]

    Parameters
    ----------
    X : 3D array: float 
        size TxMxN where T is the number of time points

    Returns
    -------
    S : 2D array: float
        Shannon Entropy matrix.

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
    identity sin**2 alpha +cos**2 alpha=1

    Parameters
    ----------
    x1 : 1D array: float
        Lenghth L.
    x2 : 1D array: float
        Length L.
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
    """ Absolute error between two signals"""
    return np.abs(x1-x2)
    
    
    
def phaseLockingValueTwoNodes(x1,x2):
    """
    Phase locking between x1 and X2

    Parameters
    ----------
    x1 : 1D array: float
        Lenghth L.
    x2 : 1D array: float
        Length L.

    Returns
    -------
    plv : 1D array
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
    X : 2D array: float
        Nodes x Time 

    Returns
    -------
    plv : 2D array
        phase locking value matrix

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
    diffX : 3D array: float
        Time X Nodes X Nodes difference between phases

    Returns
    -------
    plv : 2D array
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
    x1 : 1D array: float
        Lenghth L.
    x2 : 1D array: float
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
        percentage of coincident points

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
    δ : 1D array, optional
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
    1D float array
        array of local order parameter.

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
    x : 2D array.
        Nodes x Time.
    nperseg : int, optional
        Number of samples of the time window. The default is 4096.
    noverlap : int, optional
        Number of samples of the overlap window. The default is 3600.

    Returns
    -------
    freqs: 1D float array
        frequency bins 
    Cxx : 3D float array
        Tensor of coherences of each pair of nodes.
        Node x Node x Frequency

    """
    
    Cxx=np.zeros((np.shape(x)[0],np.shape(x)[0],nperseg//2+1))
    for i in range(np.shape(x)[0]):
        for j in range(np.shape(x)[0]):
            freqsc,Cxx[i,j,:]=signal.coherence(np.cos(x[i,:]),np.cos(x[j,:]),fs=fs,nperseg=nperseg,noverlap=noverlap)
    return freqsc, Cxx



def synchronyTwoNodes(x1,x2,n=1,m=1):
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
    #Define the filter
    if type=='butterworth':
        if f_high>fs/100:
            b,a=signal.butter(4,[2*f_low/fs,2*f_high/fs],btype='bandpass')
        else:
            b,a=signal.butter(3,[2*f_low/fs,2*f_high/fs],btype='bandpass')
    elif type=='chebysev': 
        b,a=signal.cheby1(4,1e-6,[2*f_low/fs,2*f_high/fs],btype='bandpass')
    #Zero-phase filter
    sinX=np.sin(X)
    Xf=signal.filtfilt(b,a,sinX)
    if simulated:
       Xf=Xf[:,3*fs:-3*fs]
    #Hilbert transform
    Xa=signal.hilbert(Xf,axis=1)
    angles=np.angle(Xa)
    amplitudes=np.abs(Xa)
    return amplitudes, angles

def FC_filtered(X,f_low=0.5,f_high=100,fs=1000,simulated=True):
    #Signal duration must be higher than 12 seconds, or the same np.shape(X)[1]>12fs
    #This limitation assures that the filter works well where theres is not signal in simulated data
    #Assume X is NxT
    amplitudes,angles=hilbertTheta(X,f_low=f_low,f_high=f_high,fs=fs,simulated=simulated)
    #Low-pass 0.5 Hz
    b,a=signal.butter(4,2*0.5/fs,btype='lowpass')
    envelopes=signal.filtfilt(b,a,amplitudes,axis=1)
    envelopes=envelopes[:,3*fs:-3*fs]
    mean_sd_envelopes=np.mean(np.std(envelopes,axis=1))
    N=np.shape(X)[0]
    FC=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i!=j:
                FC[i,j]=elementFunctionalConnectivity(envelopes[i,:],envelopes[j,:])
            else:
                FC[i,j]=1
    return FC, mean_sd_envelopes

def diffPhaseHilbert(X,f_low=2,f_high=100,fs=1000,simulated=True):
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
    return np.abs(x)%np.pi*180/np.pi
