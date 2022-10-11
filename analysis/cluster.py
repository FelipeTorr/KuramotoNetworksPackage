#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.abspath('../'))
import analysis.synchronization as synchronization
import metis
from networkx.algorithms.community import k_clique_communities
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

def Clustering(G,No_Clusters):
    color_map = []
    if No_Clusters>10:
        print("Please Edit the code to account for more than 10 clusters")
        No_Clusters=10
    # No_Clusters=9 # Can be assigned, But add extra colors below
    (edgecuts, parts) = metis.part_graph(G, No_Clusters,recursive=True)
    colors = ['red','blue','green','brown','yellow','black','magenta','olive','cyan','purple']
    for i, p in enumerate(parts):
        G.nodes[i]['color'] = colors[p]
        color_map.append(colors[p])
    return(G,color_map)

def significantFC(X,f_low=0.5,f_high=100,fs=1000,Nshuffles=20):
    #Assume X is NxT
    originalFC,mean_energy=synchronization.FC_filtered(X,f_low=f_low,f_high=f_high,fs=fs)
    T=np.shape(X)[1]
    N=np.shape(X)[0]
    indexes=list(np.arange(0,T))
    shuffledFC=np.zeros((N,N,Nshuffles))
    thresholdedFC=np.zeros((N,N))
    #Shufle the sampling point indexes
    for n in range(Nshuffles):
        np.random.shuffle(indexes)
        shuffledFC[:,:,n],_=synchronization.FC_filtered(X[:,indexes],f_low=f_low,f_high=f_high,fs=fs)
    #Selection of the threshold
    threshold=1.00
    for percentil in np.arange(95,80,-1,dtype=int):
        th=np.percentile(shuffledFC,percentil)
        if th<1:
            threshold=th
            break

    thresholdedFC[np.abs(originalFC)>threshold]=originalFC[np.abs(originalFC)>threshold]
    return originalFC,thresholdedFC,threshold,percentil, mean_energy

def sortMatrix(X):
    N=np.shape(X)[0]
    upper_triang=np.triu(X,k=1)
    sorted_indexes=np.flip(np.argsort(np.sum(upper_triang,axis=1)))
    sortedX=X[sorted_indexes,:][:,sorted_indexes]
    return sortedX, sorted_indexes

def hierarchyKMeans(FC):
    Z=linkage(1-FC,'average')
    return Z

