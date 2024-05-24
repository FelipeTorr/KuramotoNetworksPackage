#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:38:46 2024

@author: felipe

'DMD simplified'
"""

import numpy as np
from pydmd import DMD, BOPDMD
from pydmd.preprocessing import hankel_preprocessing
import sys
import os
sys.path.append(os.path.abspath('../'))
import analysis.control as ctrl


def argsort_eigs(sigma_svd):
    """
    Sort the eigenvalues by the imaginary part.
    The real eigenvalues come first.

    Parameters
    ----------
    sigma_svd : 1D complex array
        eigenvalues from a matrix.

    Returns
    -------
    imagsort : int array
        Sorted indexes of sigma_svd by the imaginary part.
    """
    
    real_part=np.real(sigma_svd)
    imag_part=np.imag(sigma_svd)
    only_real=np.argwhere(imag_part==0)
    imagsort=np.argsort(imag_part)
    outsort=np.zeros_like(imagsort)
    
    if only_real is not []:
        size_only_real=len(only_real[:,0])
        
        sort_reals=np.argsort(real_part[only_real[:,0]])
        realsort=only_real[sort_reals][:,0]
        outsort[0:size_only_real]=realsort
    else:
        size_only_real=0
    
    #Reorder the conjugate pairs to get positive imaginary part before the negative.
    conjugate_pairs=int((len(sigma_svd)-size_only_real)/2)
    
    for m in range(conjugate_pairs):
        if imag_part[imagsort[m]]>0:
            outsort[-(2*m+1)]=imagsort[-(m+1)]
            outsort[-(2*m+2)]=imagsort[m]
        else:
            outsort[-(2*m+1)]=imagsort[m]
            outsort[-(2*m+2)]=imagsort[-(m+1)]
                
    return outsort


def compute_rank(sigma_svd, rows, cols, svd_rank):
    """
    Rank computation for the truncated Singular Value Decomposition.
    Modified version of pydmd.utils._compute_rank.
    
    Parameters
    ----------
    sigma_svd : 1D (2D) complex array
        Eigenvalues from the matrix X.
        If sigma_svd is 2D, the first dimension is considered 
        the number of eigenvalues arrays.
    rows : int
        Number of rows of the matrix X
    cols : int
        Number of columns of the matrix X.
    svd_rank : int or float
        if svd_rank==0, the Hard Threshold is calculated. 
        The Hard threshold depends of a noise level.
        If svd_rank is an integer and >0, that number is used as the rank
        If svd_rank is a float between 0 and 1, it represents the threshold of 
        the amount of covariance/energy used to calculate the rank.

    Returns
    -------
    rank: int
        The calculated rank
    
    """
    
    if svd_rank == 0:
    #Hard threshold
        beta = np.divide(*sorted((rows, cols)))
        lambda_beta=np.sqrt(2*(beta+1)+(8*beta)/((beta+1)+np.sqrt(beta**2+14*beta+1)))
        sigma_noise=1/3 # Works well for AAL90, but requires work to determine the optimum value
        tau = lambda_beta*np.sqrt(cols)*sigma_noise
        if len(np.shape(sigma_svd))>1: #Tensor eigenvalues
            rank=np.zeros((np.shape(sigma_svd)[0],),dtype=int)
            for n in range(np.shape(sigma_svd)[0]):
                rank[n] = np.sum(sigma_svd[n,:] > tau) 
        else: #Matrix eigenvalues
            rank = np.sum(sigma_svd > tau)
    elif 0 < svd_rank < 1:
    #Threshold as amount of covariance/energy
        if len(np.shape(sigma_svd))>1: #Tensor eigenvalues
            rank=np.zeros((np.shape(sigma_svd)[0],),dtype=int)
            for n in range(np.shape(sigma_svd)[0]):
                cumulative_energy = np.cumsum(np.abs(sigma_svd[n,:])**2 / np.abs(sigma_svd[n,:]**2).sum())
                rank[n] = np.searchsorted(cumulative_energy, svd_rank) + 1
        else:
            cumulative_energy = np.cumsum(np.abs(sigma_svd)**2 / np.abs(sigma_svd**2).sum())
            rank = np.searchsorted(cumulative_energy, svd_rank) + 1
    elif svd_rank >= 1 and isinstance(svd_rank, int):
    #Rank defined explicitly
        rank = np.min([svd_rank,sigma_svd.size])
    else:
    #Maximum possible rank
        rank = np.min([rows, cols])
    return rank


def trunkSVD(X,svd_rank=0):
    """
    Returns the Singular value Decomposition  matrices 
    trunked with svd_rank eigenvalues
    
    Parameters
    ----------
    X : 2D (3D) complex array
        A matrix or 3D tensor.

    svd_rank : int or float, optional
        The rank used to trunk the SVD matrices. 
        The default is 0, then the hard threshold is calculated.
        If svd_rank is an integer and >0, that number is used as the rank
        If svd_rank is a float between 0 and 1, it represents the threshold of 
        the amount of covariance/energy used to calculate the rank.

    Returns
    -------
    trunk_U: 2D (3D) complex array 
         Matrix (matrices) of the left eigenvectors.
    trunk_s: 1D (2D) xomplex array
        Array of the eigenvalues 
    trunk_V: 2D (3D) complex array.
        Transpose conjugated matrix (matrices) of the right eigenvectors.
    """
    
    U, s, V = np.linalg.svd(X, full_matrices=False)
    if len(np.shape(X))==3:
        rank = compute_rank(s, X.shape[1], X.shape[2], svd_rank)
        global_rank=int(np.max(rank))
        trunk_U=np.zeros((X.shape[0],X.shape[1],global_rank))
        trunk_V=np.zeros((X.shape[0],global_rank,X.shape[2]))
        trunk_s=np.zeros((X.shape[0],global_rank))

        if svd_rank>0 and isinstance(svd_rank, int):
            aux_rank=rank
            rank=np.ones((X.shape[0],),dtype=int)*aux_rank
            
        for n in range(X.shape[0]):
            
            
            trunk_U[n,:,:rank[n]] = U[n,:, :rank[n]]
            trunk_V[n,:rank[n],:] = V[n,:rank[n],:]
            trunk_s[n,:rank[n]] = s[n,:rank[n]]
    else:
        rank = compute_rank(s, X.shape[0], X.shape[1], svd_rank)
        trunk_U = U[:, :rank]
        trunk_V = V[:rank,:]
        trunk_s = s[:rank]
    return(trunk_U,trunk_s,trunk_V,rank)

def TC(X):
    """
    Transpose conjugated operation of a matrix

    Parameters
    ----------
    X : 2D complex array
        A matrix.

    Returns
    -------
    X* : 2D complex array
        Transpose conjugated of X.

    """
    
    return X.conj().T

def trunc(values, decs=0):
    """
    
    Truncate decimals of real numbers.

    Parameters
    ----------
    values : float
        Real number.
    decs : int, optional
        Number of decimals. The default is 0.

    Returns
    -------
    float
        Real number with truncated decimals.

    """
    
    return np.trunc(values*10**decs)/(10**decs)


def networkDMD(x,C,M=2,u=None,drive_nodes=None,rankX=-1,rankY=None,dt=0.001,returnMatrices=True):
    
    #Binarize the adjacency matrix
    C_binary=np.zeros_like(C)
    C_binary[C>0]=1
    
    assert np.shape(C)[0]==np.shape(C)[1]==np.shape(x)[0], 'C must be a squared matrix representing a graph that generate x dynamics'
    N=np.shape(x)[0]
    Tp=np.shape(x)[1]
    T=Tp-1
    assert T>=2, 'at least two time points are needed'
    
    # If control signal is not explicit
    if u is None:
        u=np.zeros((1,Tp-1))
        L=1
        drive_nodes=[0]
    else:
        L=np.shape(u)[0]
        assert L==len(drive_nodes), 'There must be a control signal for each drive node.'
    #Rank of the SVD decompostions
    if rankX==-1 and rankY is None or rankY==-1:
        rankX=0.999
        rankY=0.9999
    elif rankY!=-1:
        rankX=0.999
        
    
    matricesbigA=np.zeros((N,N,M))
    matricesbigB=np.zeros((N,L,M))
    Y=x[:,1::]
    Z=x[:,0:-1]
    
    for m, skip in enumerate(np.arange(M)):
        #STEP 1
        #Build OMEGA for each node at each window
        Zm=Z[:,skip:T]
        lenTime=np.shape(Zm)[1]
        Gamma=np.zeros((N,N+1,lenTime))
        OMEGA=np.zeros((N,N+1,lenTime))
        
        l=0
        for n in range(N):
            for i in np.argwhere(C_binary[n,:]==1)[:,0]:
                Gamma[n,i,:]=Zm[i,:]
            if n in drive_nodes:
                Gamma[n,-1,:]=u[l,skip:T]
                l+=1
            indexes=ctrl.indexes_untilN_remove_m(N+1,n)
            OMEGA[n,:,:]=np.vstack((Zm[n:n+1,:],Gamma[n,indexes,:]))
        
        U,s,V,rank=trunkSVD(OMEGA,svd_rank=rankX)
        
        #STEP 2
        #Build the network matrix
        U1=U[:,0:1,:]
        U2=U[:,1:N,:]
        U3=U[:,N::,:]
        
        Ajj=np.zeros((N,1))
        Ajk=np.zeros((N,N-1))
        Bjl=np.zeros((N,L))
        
        bigA=np.zeros((N,N))
        
        for n in range(N):
            Ajj[n,:]=Y[n,skip:T]@TC(V[n,:rank[n],:])@np.diag(1/s[n,:rank[n]])@TC(U1[n,:,:rank[n]])
            Ajk[n,:]=Y[n,skip:T]@TC(V[n,:rank[n],:])@np.diag(1/s[n,:rank[n]])@TC(U2[n,:,:rank[n]])
            Bjl[n,:]=Y[n,skip:T]@TC(V[n,:rank[n],:])@np.diag(1/s[n,:rank[n]])@TC(U3[n,:,:rank[n]]) 
            bigA[n,n]=Ajj[n,:]
            bigA[n,ctrl.indexes_untilN_remove_m(N,n)]=Ajk[n,:]
        
        bigB=Bjl
        matricesbigA[:,:,m]=bigA
        matricesbigB[:,:,m]=bigB
        
    # STEP 3 Average Network Transition matrix
    mean_bigA=np.mean(matricesbigA,axis=2)
    mean_bigB=np.mean(matricesbigB,axis=2)
    # sd_bigA=np.std(matricesbigA,axis=2)
    # sd_bigB=np.std(matricesbigB,axis=2)
    
    UY,sY,VY,rankY=trunkSVD(Y,svd_rank=rankY)
    smallA=TC(UY)@mean_bigA@UY
    smallB=TC(UY)@mean_bigB
    eigsA,eigvectorsA=np.linalg.eig(smallA)
    
    # Continuous time eigvalues
    seigs=1/dt*np.log(eigsA) 
    
    #sort eigvalues by frequency
    sort_eigs=argsort_eigs(seigs)
    seigs=seigs[sort_eigs]
    eigsA=eigsA[sort_eigs]
    eigvectorsA=eigvectorsA[sort_eigs]
    # STEP 5
    
    # Theoretical Modes
    # PHI=mean_bigA@UY@eigvectorsA
    
    
    # Fit modes to the amplitudes
    ttp=np.arange(0,(Tp)*dt,dt)
    PHI_b=x@np.linalg.pinv(np.exp(np.outer(seigs,ttp)))


    
    #Optimal amplitudes
    # vander = np.vander(eigsA, T, True)

    # P = np.multiply(np.dot(TC(PHI), PHI),
    #                 np.conj(np.dot(vander, TC(vander))))
    # q = np.conj(np.diag(np.linalg.multi_dot([vander, TC(x[:,0:T]), PHI])))
    
    # a = np.linalg.solve(P,q)
    
    # PHI_b=PHI@np.diag(a)
    
   
    
    if returnMatrices:
        return seigs, PHI_b, eigsA, mean_bigA,mean_bigB, smallA, smallB
    else:
        return seigs, PHI_b

def sortedDMD(x,rank,dt,d=2):
    """
    Dynamic Modes Decomposition with modes sorted by the frequency

    Parameters
    ----------
    x : 2D(1D) float array
        Data of shape N channels x T time samples.
    rank : int
        Number of poles, note that each oscillatory mode is a pair of conjugate poles.
    dt : float
        Sampling period.
    d : int, optional
        Number of frames/ delays. The default is 2.

    Returns
    -------
    sorted_eigs : 1D complex array
        Eigenvalues of DMD, the poles of the LTI system (\Omega).
    sorted_amplitudes : 1D float array
        Amplitudes of DMD, the weight of each mode (b).
    sorted_modes : 2D float array
        Spatial Modes of DMD, the coefficients for the linear combination of the dynamic modes.

    """
    
    #time array withoth the last d samples 
    delay_t=np.arange(0,(np.shape(x)[1]-d+1)*dt,dt)
    N=np.shape(x)[0]
    #Initialize optimal DMD for real signals and with the desired rank
    # assert rank%2==0, 'The rank must be even to get cojugate poles'
    optdmd=BOPDMD(svd_rank=rank,num_trials=0,eig_constraints={'conjugate_pairs'})
    delay_optdmd=hankel_preprocessing(optdmd, d=d)
    delay_optdmd.fit(x,t=delay_t)
    modes_mult=delay_optdmd.modes
    eigs=delay_optdmd.eigs
    amplitudes=delay_optdmd.amplitudes
    sort_frequencies_indexes=np.argsort(np.abs(np.imag(eigs)))
    modes=np.zeros((N,rank),dtype=complex)
    for i in range(d):
        modes_iter=modes_mult[i*N:(1+i)*N,:]
        modes+=modes_iter
    modes/=d
    sorted_eigs=eigs[sort_frequencies_indexes]
    sorted_amplitudes=amplitudes[sort_frequencies_indexes]
    sorted_modes=modes[:,sort_frequencies_indexes]    
    return  sorted_eigs,sorted_amplitudes, sorted_modes

def stableDMD(x,rank,dt,d=2):
    eigs,amplitudes,modes=sortedDMD(x,rank,dt,d)
    stable_eigs_real=np.real(eigs)
    stable_eigs_imag=np.imag(eigs)
    not_stable=np.argwhere(np.real(eigs)>0)
    stable_eigs_real[not_stable]=0
    stable_eigs=stable_eigs_real+1j*stable_eigs_imag
    return stable_eigs, amplitudes,modes

def frequencies(eigs):
    freqs=np.abs(np.imag(eigs))/(2*np.pi)
    deduplicated_freqs = set(freqs)
    deduplicated_freqs = np.sort(np.array(list(deduplicated_freqs)))
    return deduplicated_freqs
    
def reconstructDMD(eigs,amplitudes,modes,t):
    x=modes@np.diag(amplitudes)@np.exp(np.outer(eigs,t))
    return x

def find_optimum_rank(x,dt,train_samples,test_samples,trials=10,minrank=1,maxrank=None,stable=True):
    import time as time
    if maxrank is None:
        maxrank=np.shape(x)[0]
    assert np.shape(x)[1]>=(train_samples+test_samples), 'Time length of x must be greater or equal to the sum of train and test samples'
    opt_rank=0
    errors=np.zeros((maxrank,trials))
    times=np.zeros((maxrank,trials))
    
    for trial in range(trials):
        x_origin_test=x[:,(trial+1)*train_samples:(trial+1)*train_samples+test_samples]
        for r in range(minrank,maxrank):
            time_1=time.time()
            if stable:
                eigs,amplitudes,modes=stableDMD(x[:,trial*train_samples:(trial+1)*train_samples],rank=r,dt=dt,d=2)
            else:
                eigs,amplitudes,modes=sortedDMD(x[:,trial*train_samples:(trial+1)*train_samples],rank=r,dt=dt,d=2)
            x_reconstruct=reconstructDMD(eigs, amplitudes, modes, t=np.arange(0,(train_samples+test_samples)*dt,dt))
            x_reconstruct_test=x_reconstruct[:,train_samples:train_samples+test_samples]
            time_2=time.time()
            times[r,trial]=time_2-time_1
            error_r=np.mean(np.abs(x_origin_test-x_reconstruct_test)**2)
            errors[r,trial]=error_r
    opt_error=np.sum(np.abs(x_origin_test)**2)
    #Average across trials
    max_errors=np.max(errors,axis=1)
    max_times=np.max(times,axis=1)
    for r in range(minrank,maxrank):
        if max_errors[r]<opt_error and max_times[r]<(train_samples*dt):
            opt_error=max_errors[r]
            opt_rank=r
    return opt_rank, opt_error, max_errors,max_times


def find_max_rank(x,dt,train_samples,test_samples,trials=10,minrank=1,maxrank=None,stable=True):
    import time as time
    if maxrank is None:
        maxrank=np.shape(x)[0]
    assert np.shape(x)[1]>=(train_samples+test_samples), 'Time length of x must be greater or equal to the sum of train and test samples'
    opt_rank=0
    errors=np.zeros((maxrank,trials))
    times=np.zeros((maxrank,trials))
    for trial in range(trials):
        x_origin_test=x[:,(trial+1)*train_samples:(trial+1)*train_samples+test_samples]
        for r in range(minrank,maxrank):
            time_1=time.time()
            if stable:
                eigs,amplitudes,modes=stableDMD(x[:,trial*train_samples:(trial+1)*train_samples],rank=r,dt=dt,d=2)
            else:
                eigs,amplitudes,modes=sortedDMD(x[:,trial*train_samples:(trial+1)*train_samples],rank=r,dt=dt,d=2)
            x_reconstruct=reconstructDMD(eigs, amplitudes, modes, t=np.arange(0,(train_samples+test_samples)*dt,dt))
            x_reconstruct_test=x_reconstruct[:,train_samples:train_samples+test_samples]
            time_2=time.time()
            times[r,trial]=time_2-time_1
            error_r=np.mean(np.abs(x_origin_test-x_reconstruct_test)**2)
            errors[r,trial]=error_r
    #Average across trials
    mean_errors=np.mean(errors,axis=1)
    mean_times=np.mean(times,axis=1)
    max_error=np.sum(np.abs(x_origin_test)**2)/(train_samples+test_samples)
    max_rank=0
    for r in range(minrank,maxrank):
        if mean_errors[r]<max_error and mean_times[r]<(train_samples*dt):
            max_rank=r
        else:
            mean_errors[r]=max_error
    return max_rank, max_error, mean_errors,mean_times

def DMD2tf(eigs,amplitudes,modes,dt):
    N=np.shape(modes)[0]
    num=np.hstack((np.real(ctrl.build_num(eigs,amplitudes,modes,dt)),np.zeros((N,1))))
    den=np.real(ctrl.build_den(eigs,dt))
    return num, den

def DMD2ss(eigs,amplitudes,modes,dt,form='canonical'):
    N=np.shape(modes)[0]
    if form=='canonical':
        A,B,C,D=ctrl.build_tf_canonical(eigs,amplitudes,modes, dt)
    elif form=='diagonal':
        A,B,C,D=ctrl.build_tf_diagonal(eigs,amplitudes,modes,dt)
    return A,B,C,D
    
    
