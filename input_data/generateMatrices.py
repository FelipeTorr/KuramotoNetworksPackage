#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 10:41:42 2022

@author: felipe
"""

import numpy as np
import scipy.io as sio
import sys
import networkx as nx


def connectionMatrix(n_nodes=4,weights_type='equal', mean_weight=1.0,std_weight=1.0,symmetric=True,positive=True):
    #weigths: equal, random, randomGaussian, user defined
    if weights_type=='equal':
        SC=np.ones((n_nodes,n_nodes))*mean_weight
        SC=SC-np.diag(np.diag(SC))
    elif weights_type=='random':
        SC=std_weight*np.random.rand(n_nodes,n_nodes)+mean_weight
        SC=SC-np.diag(np.diag(SC))
    elif weights_type=='randomGaussian':
        SC=std_weight*np.random.randn(n_nodes,n_nodes)+mean_weight
        SC=SC-np.diag(np.diag(SC))
    if symmetric:
        SC[np.tril_indices(n_nodes,-1)]=SC.T[np.tril_indices(n_nodes,-1)]
    
    if positive:
        SC=np.abs(SC)
        
    return SC
    
def storeMatrices(SC,D,name=None):
    if name is None:
        name=''
    else:
        name='_'+name
    dataSC={'C':SC}
    dataDelays={'D':D}
    sio.savemat('structural_connectivity%s.mat'%name, dataSC)
    sio.savemat('delays_matrix%s.mat'%name, dataDelays)
    
def main():
    print('Storing new connection matrices')
    print('Structural Connectivity matrix')
    n_nodes=int(input('Set the number of nodes:\n'))
    tipo=int(input('Set the type of the connections: 1:Equal, 2:Random Uniform, 3: Random Gaussian:\n'))
    symmetric_in=input('Symmetric [y/n]?:\n')
    positive_in=input('Only positives [y/n]?:\n')
    
    if tipo==1:
        weights_type='equal'
        mean_weight=float(input('Set weight value:\n'))
        std_weight=0
    elif tipo==2:
        weights_type='random'
        mean_weight=float(input('Set mean weight value:\n'))
        std_weight=float(input('Set standard deviation of weight values:\n'))
    elif tipo==3:
        weights_type='randomGaussian'
        mean_weight=float(input('Set mean weight value:\n'))
        std_weight=float(input('Set standard deviation of weight values:\n'))
    
    
    print('Delays matrix')
    tipoDelays=int(input('Set the type of the connections: 1:Equal, 2:Random Uniform, 3: Random Gaussian:\n'))
    
    
        
    if tipoDelays==1:
        weights_typeDelays='equal'
        mean_weightDelays=float(input('Set delay value:\n'))
        std_weightDelays=0
    elif tipoDelays==2:
        weights_typeDelays='random'
        mean_weightDelays=float(input('Set mean delay value:\n'))
        std_weightDelays=float(input('Set standard deviation of delay values:\n'))
    elif tipoDelays==3:
        weights_typeDelays='randomGaussian'
        mean_weightDelays=float(input('Set mean delay value:\n'))
        std_weightDelays=float(input('Set standard deviation of delay values:\n'))
    
    if symmetric_in=='y':
        symmetric=True
    else:
        symmetric=False
    
    if positive_in=='y':
        positive=True
    else:
        positive=False
        
    SC=connectionMatrix(n_nodes=n_nodes,weights_type=weights_type,symmetric=symmetric,positive=positive,mean_weight=mean_weight,std_weight=std_weight)
    D=connectionMatrix(n_nodes=n_nodes,weights_type=weights_typeDelays,mean_weight=mean_weightDelays,std_weight=std_weightDelays)

    storeMatrices(SC, D)
        
if __name__=='__main__':
    main()
    
    