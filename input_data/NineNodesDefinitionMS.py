#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 18:17:13 2022

@author: felipe
"""

import sys
import os
sys.path.append(os.path.abspath('../analysis'))
sys.path.append(os.path.abspath('../model'))
import numpy as np
import matplotlib.pyplot as plt
import connectivityMatrices

def setMatrices():
    C=np.zeros((9,9))
    C1=np.zeros((9,9))
    C2=np.zeros((9,9))
    C2=np.zeros((9,9))
    C3=np.zeros((9,9))
    D=np.zeros((9,9))
    D1=np.zeros((9,9))
    D2=np.zeros((9,9))
    D3=np.zeros((9,9))
    
    #Same coupling strengths
    
    #Red Nodes 0,1,2
    C[0,1]=1
    C[0,2]=1
    C[1,0]=1
    C[1,2]=1
    C[2,0]=1
    C[2,1]=1
    #Red to Yellow
    C[0,3]=1
    C[3,0]=1
    C[1,4]=1
    C[4,1]=1
    C[2,5]=1
    C[5,2]=1
    #Yellow Nodes 3,4,5
    C[3,4]=1
    C[3,5]=1
    C[4,3]=1
    C[4,5]=1
    C[5,3]=1
    C[5,4]=1
    #Yellow to Blue
    C[3,6]=1
    C[6,3]=1
    C[4,7]=1
    C[7,4]=1
    C[5,8]=1
    C[8,5]=1
    #Blue Nodes 6,7,8
    C[6,7]=1
    C[6,8]=1
    C[7,6]=1
    C[7,8]=1
    C[8,6]=1
    C[8,7]=1
    
    C0=np.copy(C)
    D0=np.copy(C)
    
    np.random.seed(2);
    C1[C>0]=np.abs(np.random.randn(30,)*3.2833*5/3);
    D1[C>0]=np.abs(np.random.randn(30,)*0.02979*5/3);
    
    sdC=10
    sdD=0.1
    C2[C>0]=np.abs(np.random.randn(30,)*5/3*sdC);
    D2[C>0]=np.abs(np.random.randn(30,)*5/3*sdD);
    
    return C0,D0,C1,D1,C2,D2
