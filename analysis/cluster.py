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
    """
    Clustering a graph in the quantity indicated by No_Clusters 

    Parameters
    ----------
    G : netwokx.Graph
        Graph.
    No_Clusters : int
        Number of clusters.

    Returns
    -------
    G: clustered graph
    color_map: color of the nodes in function of the pertenence to a cluster

    """
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
    return (G,color_map)

def significantFC(X,f_low=0.5,f_high=100,fs=1000,Nshuffles=20):
    """
    Threshold of the Functional Connectivity matrix by a threshold coming from 
    the distribution of surrogate FC matrices
    Parameters
    ----------
    X : 2D array
        Phase data NxT.
    f_low : float, optional
        Low frequency (Hz) of the bandpass filter. The default is 0.5.
    f_high : float, optional
        High frequency (Hz) of the bandpass filter. The default is 100.
    fs : int, optional
        Sampling rate. The default is 1000.
    Nshuffles : int, optional
        Number of surrogate matrices (caution> each repeat almost the entire the process). The default is 20.

    Returns
    -------
    originalFC : 2D float array
        Functional connectivty matrix
    thresholdedFC : 2D float array
        Thresholded FC matrix.
    threshold : float
        Threshold value.
    percentil : float
        Percentil at where the threshold is <1.
    mean_energy : float
        mean energy of the envelopes used in the FC matrix calculation.

    """
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
    """
    Sort a symmetric matrix by the sum in each row (sum over the columns)

    Parameters
    ----------
    X : 2D array
        Symmetric matrix with no information on the diagonal.

    Returns
    -------
    sortedX : 2D array
        sorted Matrix 
    sorted_indexes : 1D array
        indexes from the original matrix X to obtain the sortedX matrix.

    """
    upper_triang=np.triu(X,k=1)
    sorted_indexes=np.flip(np.argsort(np.sum(upper_triang,axis=1)))
    sortedX=X[sorted_indexes,:][:,sorted_indexes]
    return sortedX, sorted_indexes

def hierarchyKMeans(FC):
    """
    Hierarchical clustering by average mean

    Parameters
    ----------
    FC : 2D array with range [-1,1]
        Functional connectivity matrix.

    Returns
    -------
    Z : scipy.custer.hierarchy.linkage
        Clusters based in average distance.

    """
    Z=linkage(1-FC,'average')
    return Z

def categoryDistance(x,y):
    """
    returns the index j and distance of the lower distance from i element in x to j element in y
    Parameters
    ----------
    x : 1D array 
        Feature of interest.
    y : float or 1D array
        Central(kernel) values of the clusters.

    Returns
    -------
    min_distance_index : 1D int array
        Index of the element in y more near to each element in x
    min_distance : TYPE
        The minimum distance for each pair indicated by min_distance_index.

    """
    if len(y)>1:
        dist_matrix=np.zeros((len(x),len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                dist_matrix[i,j]=np.sqrt(np.abs(x[i]**2-y[j]**2))
        min_distance_index=np.argmin(dist_matrix,axis=1)
        min_distance=dist_matrix[np.argmin(dist_matrix,axis=1)]
    else:
        min_distance_index=np.zeros((len(x)))
        min_distance=np.sqrt(np.abs(x**2-y**2))
    return min_distance_index,min_distance