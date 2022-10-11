#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.abspath('../analysis'))
sys.path.append(os.path.abspath('../model'))
import numpy as np
import matplotlib.pyplot as plt
import connectivityMatrices

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

#Disconnected levels
#Red Nodes 0,1,2
C1[0,1]=1
C1[0,2]=1
C1[1,0]=1
C1[1,2]=1
C1[2,0]=1
C1[2,1]=1
#Red to Yellow
C1[0,3]=0
C1[3,0]=0
C1[1,4]=0
C1[4,1]=0
C1[2,5]=0
C1[5,2]=0
#Yellow Nodes 3,4,5
C1[3,4]=1
C1[3,5]=1
C1[4,3]=1
C1[4,5]=1
C1[5,3]=1
C1[5,4]=1
#Yellow to Blue
C1[3,6]=0
C1[6,3]=0
C1[4,7]=0
C1[7,4]=0
C1[5,8]=0
C1[8,5]=0
#Blue Nodes 6,7,8
C1[6,7]=1
C1[6,8]=1
C1[7,6]=1
C1[7,8]=1
C1[8,6]=1
C1[8,7]=1


#One stronger

#Red Nodes 0,1,2
C2[0,1]=1
C2[0,2]=1
C2[1,0]=1
C2[1,2]=1
C2[2,0]=1
C2[2,1]=1
#Red to Yellow
C2[0,3]=1
C2[3,0]=1
C2[1,4]=1
C2[4,1]=1
C2[2,5]=10
C2[5,2]=10
#Yellow Nodes 3,4,5
C2[3,4]=1
C2[3,5]=1
C2[4,3]=1
C2[4,5]=1
C2[5,3]=1
C2[5,4]=1
#Yellow to Blue
C2[3,6]=1
C2[6,3]=1
C2[4,7]=1
C2[7,4]=1
C2[5,8]=1
C2[8,5]=1
#Blue Nodes 6,7,8
C2[6,7]=1
C2[6,8]=1
C2[7,6]=1
C2[7,8]=1
C2[8,6]=1
C2[8,7]=1


#One Weaker

#Red Nodes 0,1,2
C3[0,1]=1
C3[0,2]=1
C3[1,0]=1
C3[1,2]=1
C3[2,0]=1
C3[2,1]=1
#Red to Yellow
C3[0,3]=1
C3[3,0]=1
C3[1,4]=1
C3[4,1]=1
C3[2,5]=0.1
C3[5,2]=0.1
#Yellow Nodes 3,4,5
C3[3,4]=1
C3[3,5]=1
C3[4,3]=1
C3[4,5]=1
C3[5,3]=1
C3[5,4]=1
#Yellow to Blue
C3[3,6]=1
C3[6,3]=1
C3[4,7]=1
C3[7,4]=1
C3[5,8]=1
C3[8,5]=1
#Blue Nodes 6,7,8
C3[6,7]=1
C3[6,8]=1
C3[7,6]=1
C3[7,8]=1
C3[8,6]=1
C3[8,7]=1


D[C>0.0]=1
D1[C>0.0]=1
D2[C>0.0]=1
D2[C>0.0]=1


delays=np.linspace(0.1,1.0,15)
delays1=np.flip(delays)


m=0
for i in range(9):
    for j in range(i):
        if D[i,j]==1:
            D1[i,j]=delays[m]
            D1[j,i]=delays[m]
            
            D2[i,j]=delays1[m]
            D2[j,i]=delays1[m]
        
            m+=1

description=["C: Same coupling strength", "C1: separated levels", 
"C2: connection between nodes 2,5 stronger","C3: connection between nodes 2,5 weaker",
"D: Same delay", "D1: Delays propottional to coupling", "D2: Delay innverse to coupling"]

np.savez('nine_nodes_network.npz',description=description,C=C,C1=C1,C2=C2,C3=C3,D=D,D1=D1,D2=D2)
