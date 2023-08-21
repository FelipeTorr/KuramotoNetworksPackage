#!/usr/bin/env python3
# -*- coding: utf-8 -*-
try:
    import analysis.Wavelets as Wavelets
except ModuleNotFoundError:
    import KuramotoNetworksPackage.analysis.Wavelets as Wavelets
import numpy as np
import numpy.linalg as linalg
import scipy.signal as signal
import scipy.ndimage as ndimage
#import emd


def effectiveFrequency(x,T):
    """
    Calculate the effective frequency of a phase serie.
    It is defined as (final_phase - initial_phase)/(2(pi)T)
    
    Parameters
    ----------
    x : 2D array
        Nodes x time samples.
    T : float
        Total time (seconds).

    Returns
    -------
    1D float array
        Effective frequency (Hz).

    """
    return (x[:,-2]-x[:,1])/(2*np.pi*T)
    
def peak_freqs(x,fs=1000,nperseg=4096,noverlap=2048,applySin=True,includeDC=False):
    """
    The peak of the Welch's periodogram.
    
    Parameters
    ----------
    x : 1D float
        data NxT.
    fs : int
    	sampling frequency (samples/second)
    nperseg : int, optional
        time window in number of samples. The default is 4096 samples.
    noverlap : int, optional
        overlap time in number of samples. The default is 2048 samples.
    applySin : boolean
        Apply or not the *sin()* function before calculate the spectrum. The default is True.
    includeDC : boolean, default=False
        include or not the DC component to find the peak frequency
    
    Returns
    -------
    f : 1D float array
        list of frequencies.
    Pxx : 2D float array
        Spectrograms: Nodes x frequency
    pfreqs : 1D float array
        frequency peak for each node.

    """
    
    if applySin:
        X=np.sin(x)
    else:
        X=x
    pfreqs=np.zeros((np.shape(x)[0],))
    Pxx=np.zeros((np.shape(x)[0],nperseg//2+1))
    if len(np.shape(x))>1:
        for n in range(np.shape(x)[0]):
            f,Pxx[n,:]=signal.welch(X[n,:],fs=fs,window='hamming',nperseg=nperseg,noverlap=noverlap)
            if includeDC==True:
                pfreqs[n]=f[np.argmax(Pxx[n,:])]
            else:
                pfreqs[n]=f[np.argmax(Pxx[n,1::])+1]
    else:
        #Single node
        f,Pxx=signal.welch(X,fs=fs,window='hamming',nperseg=nperseg,noverlap=noverlap)
        if includeDC==True:
            pfreqs=f[np.argmax(Pxx)]
        else:
            index_max_freq=np.argmax(Pxx[1::])+1
            if index_max_freq==0 or index_max_freq==len(Pxx):
                index_max_freq==0
            pfreqs=f[index_max_freq+1]
    return f,Pxx,pfreqs

def countSumPeaks(x,fs=1000,nperseg=4096,noverlap=2048,applySin=True,minProminence=0.5,maxProminence=1000,distance=0.5):
    """"
    The peaks of the sum of the all nodes Welch's periodograms.
    
    Parameters
    ----------
    x : 1D float
        data NxT.
    fs : int
    	sampling frequency (samples/second)
    nperseg : int, optional
        time window in number of samples. The default is 4096 samples.
    noverlap : int, optional
        overlap time in number of samples. The default is 2048 samples.
    applySin : boolean
        Apply or not the *sin()* function before calculate the spectrum. The default is True.
    minProminence: float
        minimum value of ratio between the peak and its neighborhood
    maxProminence: float
    	maximum value of ratio between the peak and its neighborhood
    distance: float
    	minimum value in Hz between peaks	
    	
    Returns
    -------
    f : 1D float array
        list of frequencies.
    sumPxx : 1D float array
        Sum of the periodograms.
    npeaks : int
        Number of found peaks
    peaks_arrayt : 2D float array
        frequencies and amplitudes of the found peaks

    """
    if applySin:
        X=np.sin(x)
    else:
        X=x
    Pxx=np.zeros((np.shape(x)[0],nperseg//2+1))
    if len(np.shape(x))>1:
        for n in range(np.shape(x)[0]):
            f,Pxx[n,:]=signal.welch(X[n,:],fs=fs,window='hamming',nperseg=nperseg,noverlap=noverlap)
        #Sum
        sumPxx=np.sum(Pxx,axis=0)
    else:
        #Single node
        f,sumPxx=signal.welch(X,fs=fs,window='hamming',nperseg=nperseg,noverlap=noverlap)
    df=f[1]-f[0]
    ndistance=int(distance//df)
    peaks, properties=signal.find_peaks(sumPxx,prominence=(minProminence,maxProminence),distance=ndistance)
    npeaks=len(peaks)
    peaks_array=np.zeros((npeaks,2))
    for i in range(npeaks):
        peaks_array[i,0]=f[peaks[i]]
        peaks_array[i,1]=sumPxx[peaks[i]]
        
    return f, sumPxx, npeaks, peaks_array
    
def peaksSpectrum(f,Pxx,Npeaks=10,deltaf=5,useScipy=True):
    """
    Find **Npeaks** in the spectrum **Npeaks**
    
    Parameters
    ----------
    f : 1D float array
        frequencies list (Hz).
    Pxx : 1D or 2D float array
        N x Spectrum.
    Npeaks : int, optional
        Number of peaks. Default is 10.
    deltaf : int, optional
        Frequency range for tolerance between peaks. The default is 5 Hz.
    useScipy : boolean, optional
        Set to use scipy method, or use the own peaks-search method
    
    Returns
    -------
    pindex : 1D int array
        Indexes of the peak frequencies in the frequencies list, f.
    pfreqs : 1D float array
        Frequency peaks.

    """
    
    df=f[1]-f[0]
    delta=int(deltaf/df)+1 
        
    if len(np.shape(Pxx))==1:
        if useScipy:
            pindex=findPeaksScipy(Pxx,tolF=delta)
        else:
            pindex=findPeaks(Pxx,tolF=delta,Nmax=Npeaks)
        pfreqs=f[pindex]
    else:
        N=np.shape(Pxx)[0]
        pindex=np.zeros((N,Npeaks))
        pfreqs=np.zeros((N,Npeaks))
        for j in range(N):
            if useScipy:
                pindex_j=findPeaksScipy(Pxx,tolF=delta)
                npeaks_j=len(pindex_j)
            else:
                pindex_j=findPeaks(Pxx,tolF=delta,Nmax=Npeaks)
                npeaks_j=Npeaks
            pindex[j,0:npeaks_j]=pindex_j    
            pfreqs[j,0:npeaks_j]=f[pindex]
    
    return pindex,pfreqs

def findPeaks(Pxx,tolPercentile=1,tolF=10,Nmax=10,power_quotient=2):
    """
    Find the peaks in the spectrum *Pxx* that accomplish:
    1. Derivative value less that **tolPercentile**(diff(**Pxx**))
    2. At least separated **tolF** bins.
    3. Amplitude value higher than max(**Pxx**)/**power_quotient**
    

    Parameters
    ----------
    Pxx : float 1D array
        Spectrum.
    tolPercentile : int, optional
        Percentil of low diff values where search for peaks. The default is 1%.
    tolF : int, optional
        Number of frequency bins that should be at least between peaks. The default is 10 bins.
    Nmax : int, optional
        Maximum number of frequency peaks to find. The default is 10.
    power_quotient : float, optional
        Inverse scaling of the maximum power that a peak must achieve to be considered as a 'good' one. The default is 2.

    Returns
    -------
    peak_indexes : int 1D array
        Array of position indexes of the peaks in **Pxx**.

    """
    
    peak_indexes=np.zeros((Nmax,),dtype=int)
    diffPxx=Pxx[1::]-Pxx[0:-1]
    absdiffPxx=np.abs(diffPxx)
    tol=np.percentile(absdiffPxx,tolPercentile)
    npeak=0
    current_f=0
    #Take the mean of three consecutive points to detect cross-zero points  
    next_f=3
    maxPxx=np.max(Pxx)
    thresholdPower=maxPxx/power_quotient
    while next_f<len(Pxx)-1 and npeak<Nmax:
        if (np.mean(diffPxx[current_f:next_f]))<tol:
            search_peak=True
            for f_index in range(current_f,next_f):
                if search_peak:
                    if Pxx[f_index+1]<Pxx[f_index]:
                        if Pxx[f_index]>=thresholdPower:
                            peak_indexes[npeak]=f_index
                            npeak+=1
                            #If there is a peak, skip the tolerance frequency
                            current_f+=tolF
                            next_f+=tolF
                            break
                        search_peak=False
            search_peak=False
        current_f+=1
        next_f+=1
    return peak_indexes

def findPeaksScipy(Pxx,tolPercentile=1,tolF=10,power_quotient=0.2):
    """
    Find the peaks in the spectrum **Pxx** that accomplish:
    1. Prominence of peaks larger than the  **tolPercentile** of the diff(**Pxx**).
    2. At least separated **tolF** bins.
    3. Amplitude value higher than **power_quotient**

    Parameters
    ----------
    Pxx : float 1D array
        Spectrum.
    tolPercentile : int, optional
        Prominence of the peaks. The default is 1.
    tolF : int, optional
        Number of frequency bins that should be at least between peaks. The default is 10 bins.
    power_quotient : float, optional
        Threshold of power between neighborhood peaks. The default is 0.2.

    Returns
    -------
    peak_indexes : int 1D array
        Array of position indexes of the peaks in **Pxx**.

    """
    diffPxx=Pxx[1::]-Pxx[0:-1]
    absdiffPxx=np.abs(diffPxx)
    tol=np.percentile(absdiffPxx,tolPercentile)
    peak_indexes,properties=signal.find_peaks(Pxx,distance=tolF,prominence=tol,threshold=power_quotient)

    return peak_indexes

def waveletMorlet(x,fs=1000, f_start=0.5,f_end=200, numberFreq=500, omega0=15, correctF=False):
    """
    Wavelet scalogram usign the Morlet mother wavelet

    Parameters
    ----------
    x : 1D float array
        timeserie data.
    fs : fs, optional
        sampling frequency. The default is 1000 Hz.
    f_start : float, optional
        Starting frequency of analysis (larger wavelet scale). 
        This is the most affected scale by the cone of influence.
        The default is 0.5 Hz.
    f_end : float, optional
        End frequency of analyisis. The default is 200 Hz.
    numberFreq : int, optional
        Number of scales (frequency bins). The default is 500.
    omega0 : float, optional
        Central wavelet frequency. Modifies the bandwidth of the scalogram.
        The default is 15 Hz.
    correctF : boolean, optional
        If True, the result is normalized by 1/scale(frequency).
        Necessary to identify peaks if the spectrum has 1/f trend.
        The default is False.

    Returns
    -------
    freqs: 1D float array 
        Equivalent frequencies (Hz)
    scales: 1D float array
        Wavelet scales
    coefs: 2D float array: freqs x len(x) 
        Scalogram coefficients (a. u.)
        Wavelet transform is calculated for each time point.

    """
	
    freqs = np.logspace(np.log10(f_start),np.log10(f_end),numberFreq)
    dt=1/fs
    wavel=Wavelets.Morlet(x,freqs=freqs,dt=dt,omega0=omega0)
    
    coefs=wavel.getnormpower()
    scales=wavel.getscales()
    if correctF:
        for col in range(np.shape(coefs)[1]):
            coefs[:,col]=coefs[:,col]*freqs[:]
    return np.flip(freqs),np.flip(scales),np.flipud(coefs)

def spectrogram(X,fs=1000,nperseg=4096,noverlap=2048):
    """
    Welch spectrogram usign the welch periodograms

    Parameters
    ----------
    X : 1D or 2D float array
        timeseries data. If 2D: N x T
    fs : int
    	sampling frequency
    nperseg : int, optional
        time window in number of samples. The default is 4096.
    noverlap : int, optional
        overlap time in number of samples. The default is 2048.
    
    Returns
    -------
    t: 1D float array 
        center of the time window.
    f: 1D float array
        frequencies list.
    Sxx: 2D or 3D float array: N x len(f) x len(t) 
        Spectrogram with spectral power density units (x^2/Hz). 
    """
    t,f,Sxx=signal.spectrogram(X,fs=fs,window='hamming',nperseg=nperseg,noverlap=noverlap,scaling='density')
    	         
    return t,f,Sxx

# def empiricalModeDecomposition(x,fs=1000,f_start=0.2,f_end=200,numberFreq=500):
#     #Empirical decomposition (ortogonal signals)
#     imf = emd.sift.sift(x)
#     #Hilbert-Huang
#     freq_range = (f_start, f_end, numberFreq)
#     IP, IF, IA = emd.spectra.frequency_transform(imf, fs, 'hilbert')
#     hht_f, hht=emd.spectra.hilberthuang(IF, IA, freq_range, mode='amplitude', sum_time=False)
#     hht = ndimage.gaussian_filter(hht, 1)
    
#     return imf, hht_f, hht

def ARparameters(x,P=2):
    """
    Estimation of the parameters of an Auto-regressive process
    
    Parameters
    ----------
    x : 1D float array
        time serie.
    P : int, optional
        Number of AR parameters (number of poles, then it is strictly related to peaks). The default is 2.
    
    Returns
    -------
    a : float array
        Parameters of the AR process.
        These are the coeficients from the denominator of a discrete time impulse response 1/[1 a1 a2 a3]
    """
    
    correlation=np.correlate(x, x,mode='full')
    nlags=P
    Nhalf=len(x)-1
    #Positive lags
    correlation=correlation[Nhalf:Nhalf+nlags]
    #Used to normalize and get r0=1
    r0=correlation[0]
    norm_correlation=correlation/r0
    #correlation vector
    corrvect=norm_correlation[1:nlags] #after normalization
    #Not needed, but this avoids confusion
    norm_correlation=norm_correlation[0:nlags-1]
    corrmtx=np.zeros((nlags-1,nlags-1))
    #Bulid the correlation matrix
    for i in range(nlags-1):
        for j in range(nlags-1):
            corrmtx[i,j]=norm_correlation[np.abs(i-j)]        

    #Parameters (caution a0 is not obtained as it is always 1)
    a=linalg.solve(corrmtx,corrvect)
    
    return a

def ARpsd(a,worN=1000,fs=100,sigma=1e-3):
    """
    Power spectral density from AR parameters

    Parameters
    ----------
    a : 1D float array
        AR parameters [a1, a2, a3, ..., ap].
    worN : int, optional
        Number of frequency bins. The default is 1000.
    fs : int, optional
        sampling frequency, used only to scale. The default is 100.
    sigma : float, optional
        standard deviation of the noise in the AR process. The default is 1e-3.

    Returns
    -------
    psd : float 1D array
    Power spectral density of large worN
    """
    
    den = np.zeros(worN, dtype=complex)
    den[0] = 1.+0j
    for k in range(0, len(a)):
        den[k+1] = a[k]
    denf=np.fft.fft(den,worN*2)
    psd=sigma*fs/np.abs(denf)**2
    psd=np.real(psd[worN::])
    
    return psd

def spectralEntropy(Pxx):
    """ 
    Calculates the spectral entropy from the spectrum **Pxx**.
    **Pxx** could also be an array of spectrums.
    
    Parameters
    ----------
    Pxx : float 1D (2D) array
        Spectrum or spectrums' array N x frequency bins
    
    Returns
    -------
    H : float (1D array)
        Spectral entropy
    """
    H=0
    if Pxx.sum()==0:
        return H
    else:
        if len(np.shape(Pxx))==2:
            Nfreq=np.shape(Pxx)[1]
            ProbPxx=Pxx/Pxx.sum(keepdims=1)+1e-21
            H=(np.sum(-ProbPxx*np.log(ProbPxx),axis=1))/np.log(Nfreq)
        else:
            Nfreq=len(Pxx)
            ProbPxx=Pxx/Pxx.sum()+1e-21
            H=(np.sum(-ProbPxx*np.log(ProbPxx)))/np.log(Nfreq)
        return H

def spectralEntropy2D(Cxx):
    """ 
    Calculates the spectral/correlation entropy of a coherence/correlation matrix
    
    
    Parameters
    ----------
    Cxx : float 2D array
        Coherence or correlation matrix frequency bins x frequency bins
    
    Returns
    -------
    H : float
        Spectral/correlation entropy
    """

    N=np.shape(Cxx)[0]*np.shape(Cxx)[1]
    ProbCxx=Cxx/Cxx.sum()
    H=(np.sum(-ProbCxx*np.log(ProbCxx)))/np.log(N)
    return H


def spectralComplexity(Pxx,communities=None):
    """ 
    Calculates the spectral complexity from the spectrum **Pxx**.
    It uses the spectral entropy as entropies to calculate the information complexity
    **Pxx** could also be an array of spectrums.
    
    Parameters
    ----------
    Pxx : float 1D (2D) array
        Spectrum or spectrums' array N x frequency bins
    communities: dict
        Dictionary with the nodes grouped by communities.
    Returns
    -------
    C : float (1D array)
        Spectral complexity
    """
    H=spectralEntropy(Pxx)
    N=np.shape(Pxx)[0]
    C=0
    if communities==None:
        for n in range(N):
            C+=H[n]-(n+1)/N*np.sum(H)
    else:
        n=len(communities.keys())
        for k,key in enumerate(communities.keys()):
            C+=np.sum(H[communities[key]])-(k+1)/n*np.sum(H)
    return C