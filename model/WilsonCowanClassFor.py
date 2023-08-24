#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 10:45:28 2023

@author: felipe

#WC-Kuramoto
"""

import numpy as np
import matplotlib.pyplot as plt

from numba import jit,float64, vectorize
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

#Sigmoid, derivative and inverse
@vectorize([float64(float64,float64,float64)],nopython=True)
def sigmoid(x,a,c):
    return (1+np.exp(-a*(x-c)))**-1

@vectorize([float64(float64,float64,float64)],nopython=True)
def dsigmoid(x,a,c):
    return a*sigmoid(x,a,c)*(1-sigmoid(x,a,c))

@vectorize([float64(float64,float64,float64)],nopython=True)
def isigmoid(fx,a,c):
    return -(1/a)*(np.log(1-fx)-np.log(fx))+c


@jit(float64[:,:](float64,float64[:,:],float64[:,:],float64,float64[:]),nopython=True)
def wilsonCowanDet(t,X,CM,G,parameters):
    E,I = X
    tauE=parameters[0]
    tauI=parameters[1]
    a_ee=parameters[2]
    a_ei=parameters[3]
    a_ie=parameters[4]
    a_ii=parameters[5]
    P=parameters[6]
    Q=parameters[7]
    sigma=parameters[8]
    mu=parameters[9]
    rE=parameters[10]
    rI=parameters[11]
    return np.vstack(((-E + (1-rE*E)*sigmoid(a_ee*E - a_ei*I + G*np.dot(CM,E) + P,sigma,mu))/tauE,
                      (-I + (1-rI*I)*sigmoid(a_ie*E - a_ii*I + Q,sigma,mu))/tauI))

def SimulateTrans(CM,G,parameters,TSim,dtSim,E0=0,I0=0):
    """
    Runs a simulation of timeTrans. 
    """
    
    N=np.shape(CM)[0] 
    Var=np.array([E0,I0])[:,None]*np.ones((1,N))

    # generate the time vector
    timeTrans=np.arange(0,TSim,dtSim)    
    wilsonCowanDet.recompile()
    for i,t in enumerate(timeTrans):
        Var+=dtSim*wilsonCowanDet(t,Var,CM,G,parameters)
    return Var

def Simulate(CM,G,parameters,TTrans,TSim,dtSim,dt,E0=0,I0=0,verbose=False):
    """
    Runs simulation of Tsim seconds at dtSim integration step,
    and store it with sample rate of dt. 
    """
    #CM must be a square matrix
    if CM.shape[0]!=CM.shape[1]:
        raise ValueError("check CM dimensions (",CM.shape,")")
    
    if CM.dtype is not np.dtype('float64'):
        try:
            CM=CM.astype(np.float64)
        except:
            raise TypeError("CM must be of numeric type, preferred float")
    N=np.shape(CM)[0]
    Var=SimulateTrans(CM,G,parameters,TTrans,dtSim,E0,I0)
    
    # generate the time vector
    timeSim=np.arange(0,TSim,dtSim)   
    time=np.arange(0,TSim,dt)
    downsamp=int(dt/dtSim)
    Y_t=np.zeros((len(time),2,N))  #Vector para guardar datos
    # wilsonCowanDet.recompile()
    for i,t in enumerate(timeSim):
        if i%downsamp==0:
            Y_t[i//downsamp]=Var
        if t%10==0:
            print("%g of %g s"%(t,TSim))
        Var+=dtSim*wilsonCowanDet(t,Var,CM,G,parameters)
    return Y_t,time

def steadyState(CM,G,parameters,E0,I0):
    return E,I

##test
##parameters=[tauE,tauI,a_ee,a_ei,a_ie,a_ii,P,Q,sigma,mu,rE,rI]
# parameters=np.array([0.01,0.01,10,10,12,-2,0,-6,1.0,1.0,0.001,0.001],dtype=np.float64)
# G=0.01
# E0=0.5
# I0=0.5
# CM=np.array([[0,0.5,0.8],[0.5,0,0.6],[0.8,0.6,0]],dtype=np.float64)
# TTrans=5
# TSim=20
# dtSim=1e-4
# dt=1e-3

# X,time=Simulate(CM,G,parameters,TTrans,TSim,dtSim,dt,E0,I0,verbose=True)
