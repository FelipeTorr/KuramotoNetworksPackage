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

def sortedDMD(x,rank,dt,d=2):
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
    
    