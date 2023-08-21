#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### Connectivity matrix utilities
import numpy as np
import numpy.linalg as linalg
from scipy.io import loadmat
import networkx as nx

def loadConnectome(No_nodes, filename='../input_data/AAL_matrices.mat', field='C'):
    """
    Load a square connectivity matrix

    Parameters
    ----------
    No_nodes : int
        Number of nodes to load.
        
    filename : String, optional
        The **filename**, including the directory, where the connection matrix is stored. 
        The default is the AAL90 filename    

    field: String, optional
        The *file* of the **filename** that contains the connection weights matrix.
        The default is 'C'.
        
    Returns
    -------
    C : 2D float array
        Connectome matrix.

    """
    C = loadmat(filename)[field]
    assert np.shape(C)[0]==np.shape(C)[1], "Expected a square matrix for the connectome"
        
    max_No_nodes=np.shape(C)[0]
    if No_nodes>max_No_nodes:
        print('Max No. Nodes is %d'%max_No_nodes)
        No_nodes=max_No_nodes

    n = No_nodes
    C=C[:No_nodes,:No_nodes]
    C[np.diag(np.ones(n))==0] /= C[np.diag(np.ones(n))==0].mean()
    
    return C

def loadDelays(No_nodes,filename='../input_data/AAL_matrices.mat',field='D'):
    """
    Load the matrix of delays between nodes
    
    Parameters
    ----------
    No_nodes : int
        Number of nodes to load.
    
    filename : String, optional
        The **filename**, including the directory, where the delay matrix is stored. The default is the AAL90 filename    

    field: String, optional
        The *file* of the **filename** that contains the delays matrix. The default is 'D'.
    Returns
    -------
    D : 2D float array
        Matrix of delays, unit: seconds. (Or could be meters, if the **mean_delay** parameter has unit second/meter.)

    """
    D = loadmat(filename)[field]
    assert np.shape(D)[0]==np.shape(D)[1], "Expected a square matrix for the connection delays" 
    D=D[:No_nodes,:No_nodes]
    D /= 1000 # Distance matrix in meters 
    #or seconds with conduction velocity=1 m/s
    
    return D

def loadLabels(filename='../input_data/AAL_labels.mat', field='label90'):
    """
    Load labels of the graph nodes
    
    Parameters
    ----------
    filename : String, optional
        The **filename**, including the directory, where the labels are stored. The default is '../input_data/AAL_labels.mat'.
    field : String, optional
        The name of the *file* inside 'filename' that contains the labels. 
        The default is 'label90' for the AAL90 connectivity matrix.
        
    Returns
    -------
    labels : String array or list
        Physiologycall related names of the oscillatory nodes.

    """
    
    file=loadmat('filename')
    labels=file[field]
    return labels


def constructErdosRenyiConnectome(No_nodes,p=1,seed=2):
    """
    Construct a random connected network
    
    Parameters
    ----------
    No_nodes : int
        Number of nodes.
    p : float, optional
        Degree of connectivity. The default is 1, corresponds to fully-connected network.
        A value of 0, is a empty connectome (full of zeros).

    Returns
    -------
    C : 2D float array (boolean)
        Adjacency matrix with the defined connectivity degree for each node.

    """

    C_nx=nx.erdos_renyi_graph(n=No_nodes, p=p,seed=seed)
    C=nx.to_numpy_array(C_nx)
    return C 

def applyMeanDelay(D,C,mean_delay=1.0):
    """
    Apply the mean_delay scaling to the matrix of disatances **D**.
    Take in account for the mean of **D** only the nonzero elements of C.
    
    Parameters
    ----------
    D : float, 2D array
        Distances matrix (meters).
    C : float, 2D array
        Structural connectivity matrix (a. u.).
    mean_delay : float, optional
        The **mean_delay** . The default is 1.0.

    Returns
    -------
    D : TYPE
        DESCRIPTION.

    """
    
    meanD=np.mean(D[C>0])
    D=D/meanD*mean_delay
    return D

def degreeMatrix(C):
    """
    Diagonal degree matrix.
    
    Parameters
    ----------
    C : 2D float array
        Conectome or Adjacency matrix.
    Returns
    -------
    degree_matrix : 2D float array
        Diagonal matrix.

    """    
    degree_matrix=np.zeros_like(C)
    if np.shape(C)[0]==np.shape(C)[1]:
        degree_matrix=np.diag(np.sum(C,axis=1))
    else:
        print("C must be a square matrix")
    return degree_matrix

def adjacencyMatrix(C):
    """
    Boolean logic in a 2D float array.
    
    Parameters
    ----------
    C : 2D float array
        Conectome or adjacency matrix.

    Returns
    -------
    A : 2D int array
        Boolean adjacency matrix, 1 if c[i,j]!=0.

    """
    
    A=np.zeros_like(C)
    if np.shape(C)[0]==np.shape(C)[1]:
        A[np.nonzero(C)]=1
    else:
        print("C must be a square matrix")
    return A

def intensities(C):
    """
    Sum of the rows of the connectivity matrix.

    Parameters
    ----------
    C : 2D float array
        Conectivity matrix.

    Returns
    -------
    intensities : 1D float array
        sum of the rows.

    """
    intensities=np.sum(C,axis=1)
    return intensities

def booleanDegree(C):
    """
    Sum of the rows of the boolean adjacency matrix.
    
    Parameters
    ----------
    C : 2D float array
        Conectivity matrix.

    Returns
    -------
    k_i : 1D int array
        degree of each node.
    """
    
    A=adjacencyMatrix(C)
    k_i=np.sum(A,axis=1)
    return k_i

def Laplacian(C,asAdjacency=False):
    """
    Laplacian matrix of a graph from the connectivity matrix.

    Parameters
    ----------
    C : 2D float array
        Connectivity matrix.
    asAdjacency: boolean
        If True, uses the binary adajcency matrix. Then, it matters the connection existence, and it not matters the weight.
    Returns
    -------
    L : 2D float array
        Laplacian matrix.

    """
    if asAdjacency:
        A=adjacencyMatrix(C)
    else:
        A=np.copy(C)
    D=degreeMatrix(A)
    L=D-A
    return L

def Transition(C, asAdjacency=False):
    """
    Trasition matrix from the connectivity matrix.

    Parameters
    ----------
    C : 2D float array
        Connectivity matrix.
    asAdjacency: boolean
        If True, uses the binary adajcency matrix. Then, it matters the connection existence, and it not matters the weight.
    Returns
    -------
    T : 2D float array
        Transition matrix.

    """
    if asAdjacency:
        A=adjacencyMatrix(C)
        degrees=np.sum(degreeMatrix(C),axis=1)
    else:
        A=np.copy(C)
        degrees=intensities(C)
    N=np.shape(C)[0]
    T=np.zeros_like(C)
    for n in range(N):
        if degrees[n]!=0:
    	    T[n,:]=A[n,:]/degrees[n]
        else:
    	    T[n,:]=A[n,:]/(degrees[n]+1)
    return T
    
def eigen(C,zero_threshold=1e-9):
    """
    Calculate and sort the eigenvalues and eigenvectors of the matrix C.
    
    Parameters
    ----------
    C : float 2D square array
        Connectivity or Laplacian matrix.
    zero_threshold: float. Default value 1e-9
        Threshold for smaller eigenvalues. Any eigenvalue with absoulute value lower than zero_threshold is considered a zero eigenvalue.
    Returns
    -------
    eig_values : 1D complex array
        sorted eigenvalues of C.
    eig_vectors : 2D complex array
        eigenvectors of C sorted by eigenvalues.
    algebraic_connectivty : float
        Second eigenvalue of C.
    con_comp: int
        Connected components of the network (Quantity of zero eigenvalues)

    """
    
    eig_values,eig_vectors=linalg.eig(C)
    abs_eigs=np.abs(eig_values)
    sort_index=np.argsort(abs_eigs)
    #Sort eigenvalues
    eig_values=eig_values[sort_index]
    eig_vectors=eig_vectors[:,sort_index]
    
    #count zero eigvalues
    #Algebraic connectivity and connected components
    connected_components=len(np.argwhere(np.abs(eig_values)<zero_threshold))
    if connected_components>0:
        algebraic_connectivity=np.abs(eig_values[connected_components])
    else:
        algebraic_connectivity=0
    
    return eig_values, eig_vectors, algebraic_connectivity, connected_components
    
def vonNewmanDensity(L,beta=1):
    eig_values, eig_vectors, count_zeros_eigvalues, algebraic_connectivity, con_comp=eigen(L)
    rho=np.exp(-beta*L)/np.trace(np.exp(-beta*L))
    return rho

def vonNewmanEntropy(L,beta=1):
    S=-np.trace(vonNewmanDensity(L,beta)*np.log(vonNewmanDensity(L,beta)))
    return S

def vonNewmanRelativeEntropy(L,J,beta=1):
    S=np.trace(vonNewmanDensity(L,beta)*(np.log(vonNewmanDensity(L,beta))-np.log(vonNewmanDensity(J,beta))))
    return S
    
def orthoMatrix(eig_value,eig_vector):
    matrix=np.exp(-eig_value)*np.matmul(eig_vector,eig_vector.T)
    return np.real(matrix)
    
