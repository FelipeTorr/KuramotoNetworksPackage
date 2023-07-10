#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.abspath('../'))
try:
    import analysis.synchronization as synchronization
    import analysis.connectivityMatrices as connectivityMatrices
except ModuleNotFoundError:
    import KuramotoNetworksPackage.analysis.synchronization as synchronization
    import KuramotoNetworksPackage.analysis.connectivityMatrices as connectivityMatrices
import metis
from networkx.algorithms.community import k_clique_communities
from networkx.algorithms.community import greedy_modularity_communities
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import matplotlib.pyplot as plt

def cos_similarity(matrix,N=90):
    normaliz=np.linalg.norm(matrix,axis=1)+1e-16
    occurrence_matrix=np.copy(matrix)*0
    for col in range(N):
        suma_col=np.sum(matrix[:,col])
        if suma_col>0:
            occurrence_matrix[:,col]=matrix[:,col]/normaliz
    similarity_matrix=np.matmul(occurrence_matrix,occurrence_matrix.T)
    np.fill_diagonal(similarity_matrix,0)
    return similarity_matrix

def cluster_colors():
    """
    A list of colors to distiguish the clusters

    Returns
    -------
    colors: list
        The list of colors.

    """
    return ['red','blue','yellow','cyan','green','brown','black','magenta',
          'olive','purple','orange','lightblue','lightgreen',plt.cm.tab10(0),
          plt.cm.tab10(1),plt.cm.tab10(2),plt.cm.tab10(3),plt.cm.tab10(4),
          plt.cm.tab10(5),plt.cm.tab10(6),plt.cm.tab10(7),plt.cm.tab10(8),
          plt.cm.tab10(9),plt.cm.Set1(0),plt.cm.Set1(1),plt.cm.Set1(2),plt.cm.Set1(3),
          plt.cm.Set1(4),plt.cm.Set1(5),plt.cm.Set1(6),plt.cm.Set1(7),plt.cm.Set1(8),
          plt.cm.Set2(0),plt.cm.Set2(1),plt.cm.Set2(2),plt.cm.Set2(3),
          plt.cm.Set2(4),plt.cm.Set2(5),plt.cm.Set2(6),plt.cm.Set3(0),
          plt.cm.Set3(1),plt.cm.Set3(2),plt.cm.Set3(3),plt.cm.Set3(4),
          plt.cm.Set3(5),plt.cm.Set3(6),plt.cm.Set3(7),plt.cm.Set3(8),
          plt.cm.Set3(9),plt.cm.Set3(10),plt.cm.Set3(11),plt.cm.tab20b(0),
          plt.cm.tab20b(1),plt.cm.tab20b(2),plt.cm.tab20b(3),plt.cm.tab20b(4),
          plt.cm.tab20b(5),plt.cm.tab20b(6),plt.cm.tab20b(7),plt.cm.tab20b(8),
          plt.cm.tab20b(9),plt.cm.tab20b(10),plt.cm.tab20b(11),plt.cm.tab20b(12),
          plt.cm.tab20b(13),plt.cm.tab20b(14),plt.cm.tab20b(15),plt.cm.tab20b(16),
          plt.cm.tab20b(17),plt.cm.tab20b(18),plt.cm.tab20b(19),plt.cm.tab20c(0),
          plt.cm.tab20c(1),plt.cm.tab20c(2),plt.cm.tab20c(3),plt.cm.tab20c(4),
          plt.cm.tab20c(5),plt.cm.tab20c(6),plt.cm.tab20c(7),plt.cm.tab20c(8),
          plt.cm.tab20c(9),plt.cm.tab20c(10),plt.cm.tab20c(11),plt.cm.tab20c(12),
          plt.cm.tab20c(13),plt.cm.tab20c(14),plt.cm.tab20c(15),plt.cm.tab20c(16),
          plt.cm.tab20c(17),plt.cm.tab20c(18),plt.cm.tab20c(19)]

def Clustering(G,No_Clusters):
    """
    Clustering a graph **G** in the quantity indicated by **No_Clusters** by the METS algorithm

    Parameters
    ----------
    G : netwokx.Graph
        Graph.
    No_Clusters : int
        Number of clusters, **M**.

    Returns
    -------
    G: networkx.Graph
        Clustered graph
    color_map: list
        Color of the nodes in function of the pertenence to a cluster.

    """
    color_map = []
    if No_Clusters>10:
        print("Please Edit the code to account for more than 10 clusters")
        No_Clusters=10
    # No_Clusters=9 # Can be assigned, But add extra colors below
    (edgecuts, parts) = metis.part_graph(G, No_Clusters,recursive=True)
    colors = cluster_colors()
    for i, p in enumerate(parts):
        G.nodes[i]['color'] = colors[p]
        color_map.append(colors[p])
    return (G,color_map)

def metisClustering(G,M=2):
    """
    Clustering a grap **G** with the METIS algorithm.

    Parameters
    ----------
    G : networkx.Graph
        Graph.
    M : int
        The number of requested clusters

    Returns
    -------
    clusters : dict
        A dictionary with colors as key values, and the indexes of each cluster as values.
    sort_indexes: int list
        The list of the node indexes sorted to join all the elements in each cluster.
    labels: int 1D array
        The cluster label given to each node of the graph **G**. 
    """

    labels=np.zeros((len(G.nodes),))
    (edgecuts, parts) = metis.part_graph(G, M,recursive=True)
    parts=np.array(parts)
    colors=cluster_colors()
    communities=[]
    for nn in range(M):
        communities.append(np.where(parts==nn)[0])
    clusters={}
    sort_indexes=[]
    for n_label, color,com in zip(range(M),colors,communities):
        clusters[color]=(sorted(com))
        labels[sorted(com)]=n_label
        for indx in sorted(com):
            sort_indexes.append(indx)
    
    return clusters, sort_indexes, labels

def greedyModularityClustering(G, resolution=1):
    """
    Clustering a grap **G** with the Greedy Modularity algorithm.

    Parameters
    ----------
    G : networkx.Graph
        Graph.
    resolution : float
        The resolution employed to ponderate ratio of the external connections with the intra-cluster connections. 

    Returns
    -------
    clusters : dict
        A dictionary with colors as key values, and the indexes of each cluster as values.
    sort_indexes: int list
        The list of the node indexes sorted to join all the elements in each cluster.
    labels: int 1D array
        The cluster label given to each node of the graph **G**. 
    M : int
        The number of clusters
    """
    communities=greedy_modularity_communities(G,weight='weight',resolution=resolution)
    clusters={}
    sort_indexes=[]
    labels=np.zeros((len(G.nodes),))
    colors=cluster_colors()
    for n_label,color,com in zip(range(len(communities)),colors,communities):
        clusters[color]=(sorted(com))
        labels[sorted(com)]=n_label
        for indx in sorted(com):
            sort_indexes.append(indx)
    M=len(communities)

    return clusters, sort_indexes, labels, M

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
    X : float 2D array
        Symmetric matrix with no information on the diagonal.

    Returns
    -------
    sortedX : float 2D array
        sorted Matrix 
    sorted_indexes : int 1D array
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
    FC : float 2D array 
        Functional connectivity matrix. The range of its values is [-1,1].

    Returns
    -------
    Z : scipy.custer.hierarchy.linkage
        Clusters based in average distance.

    """
    Z=linkage(1-FC,'average')
    return Z

def categoryDistance(x,y):
    """
    Returns the index **j** and distance of the lower distance from **i** element in **x** to **j** element in y
    
    Parameters
    ----------
    x : float 1D array 
        Feature of interest.
    y : float or 1D array
        Central(kernel) values of the clusters.

    Returns
    -------
    min_distance_index : int 1D array
        Index of the element in **y** more near to each element in **x**
    min_distance : float
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

def spectralBisection(L, trisection=False):
    """
    Performs the spectral bisection of a Laplacian matrix

    Parameters
    ----------
    L : float 2D array
        Laplacian matrix
    trisection: boolean
        Defines if the algorithm need to return tree clusters (in order to obtain odd **M**). The default value is False.
    Returns
    -------
    cluster0 : int 1D array
        List of indexes that correspond to the first cluster.
    cluster1 : int 1D array
        List of indexes that correspond to the second cluster.
    cluster2 : int 1D array (if trisection==True)
        List of indexes that correspond to the third cluster.
    """
    eig_values, eig_vectors, count_zeros_eigvalues, algebraic_connectivty, con_com=connectivityMatrices.eigen(L)
    real_fiedler_vector=np.real(eig_vectors[:,con_com])
    
    if trisection:
        th=np.real(eig_vectors[0,0])
        cluster0=np.argwhere(real_fiedler_vector>th)[:,0]
        cluster1=np.argwhere(real_fiedler_vector<-th)[:,0]
        cluster2=np.argwhere((real_fiedler_vector>=(-th)) & (real_fiedler_vector<=th))[:,0]
        return cluster0, cluster1, cluster2
    else:
        cluster0=np.argwhere(real_fiedler_vector>=0)[:,0]
        cluster1=np.argwhere(real_fiedler_vector<0)[:,0]
        return cluster0, cluster1
    
def clusteringSpectral(C,M=2):
    """
    Clusters obtained by spectral bisection

    Parameters
    ----------
    C : float 2D array 
        A connectivity matrix of the graph G:
    N : int
        Number of clusters, **M**. The default value is 2. 


    Returns
    -------
    clusters_post : list of int 1D arrays
        A list of the indexes of each cluster.
    """
    num_nodes=np.shape(C)[0]
    all_nodes=np.arange(num_nodes)
    
    if M==1:
        return all_nodes
    elif M==2:
        flag_odd=False
        iter_num=1
    elif M==3:
        flag_odd=True
        iter_num=1
    else:
        flag_odd=False
        iter_num=int(M//2)
        

    L=connectivityMatrices.Laplacian(C)
    clusters_pre=[]
        
    clusters_pre.append(all_nodes)
    for ii in range(iter_num):
        clusters_post=[]
        for cluster in clusters_pre:
            if ii==iter_num-1 and flag_odd:
                cluster0,cluster1,cluster2=spectralBisection(L[cluster,:][:,cluster],trisection=True)
                clusters_post.append(cluster[cluster0])
                clusters_post.append(cluster[cluster1])
                clusters_post.append(cluster[cluster2])
                flag_odd=False
            else:
                cluster0,cluster1=spectralBisection(L[cluster,:][:,cluster],trisection=False)
                clusters_post.append(cluster[cluster0])
                clusters_post.append(cluster[cluster1])
                
        clusters_pre=clusters_post
    return clusters_post

def get_subnet_features(subnet,interest_indexes):
    #Size
    size_=np.sum(subnet)
    #A network is symetric if it has the pair of nodes from each hemisphere
    n_node_index=0
    n_posible_pairs=0
    sum_impairs=0
    sum_pairs=0
    is_symmetric=False
    while n_node_index < (len(interest_indexes)-1):
        #If the nodes of interes have a pair from both hemispheres
        if interest_indexes[n_node_index]%2==0 and interest_indexes[n_node_index+1]==interest_indexes[n_node_index]+1:
            #check
            n_posible_pairs+=1
            if np.sum(subnet[n_node_index:n_node_index+2])==2:
                sum_pairs+=1
        else:
            sum_impairs+=subnet[n_node_index]
        n_node_index+=1
    if sum_pairs==n_posible_pairs and sum_impairs==0:
        is_symmetric=True
    #
    return size_, sum_pairs