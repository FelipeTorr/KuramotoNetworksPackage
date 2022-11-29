#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### Connectivity matrix utilities
import numpy as np
import numpy.linalg as linalg
from scipy.io import loadmat
import networkx as nx

def loadConnectome(No_nodes, filename='../input_data/AAL_matrices.mat', field='C'):
    """
    Load a square matrix

    Parameters
    ----------
    No_nodes : int
        Number of nodes to load.
        
    filename : String, optional
        The filename where the connection matrix is stored. The default is the AAL90 filename    

    field: String, optional
        The filed of file of the filename that contains the connections matrix.
        
    Returns
    -------
    C : 2D float array
        Conectome matrix.

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
        The filename where the delay matrix is stored. The default is the AAL90 filename    

    field: String, optional
        The filed of file of the filename that contains the delays matrix.
    Returns
    -------
    D : 2D float array
        Matrix of delays, unit: seconds.

    """
    D = loadmat(filename)[field]
    assert np.shape(D)[0]==np.shape(D)[1], "Expected a square matrix for the connection delays" 
    D=D[:No_nodes,:No_nodes]
    D /= 1000 # Distance matrix in meters 
    #or seconds with conduction velocity=1 m/s
    
    return D

def loadLabels(filename=None, field=None):
    """
    Parameters
    ----------
    filename : String, optional
        The filename where the labels are stored. The default is None.
    field : String, optional
        The field o file inside 'filename' that contains the labels. The default is None.

    Returns
    -------
    labels : String array or list
        Physiologycall related names of the oscillating nodes.

    """
    #Default values for AAL90
    if filename==None:
        filename='../input_data/AAL_labels.mat'
    if field==None:
        field='label90'
    
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
    meanD=np.mean(D[C>0])
    D=D/meanD*mean_delay
    return D

def degreeMatrix(C):
    """
    Diagonal degree matrix
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
        for i in range(np.shape(C)[0]):
            degree_matrix[i,i]=np.sum(C[i,:])
    else:
        print("C must be a square matrix")
    return degree_matrix

def adjacencyMatrix(C):
    """
    Boolean logic in a 2D float array
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
    Sum of the rows of the connectivity matrix

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
    Sum of the rows of the boolean adjacency matrix
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

def Laplacian(C,asAdjancency=False):
    """
    Lplacian matrix of a graph from the connectivity matrix

    Parameters
    ----------
    C : 2D float array
        Connectivity matrix.

    Returns
    -------
    L : 2D float array
        Laplacian matrix.

    """
    if asAdjancency:
        A=adjacencyMatrix(C)
    else:
        A=np.copy(C)
    D=degreeMatrix(A)
    L=D-A
    return L

def eigen(C):
    """
    Parameters
    ----------
    C : 2D float square array
        Connectivity or Laplacian matrix.

    Returns
    -------
    eig_values : 1D complex array
        sorted eigenvalues of C.
    eig_vectors : 2D complex array
        eigenvectors of C sorted by eigenvalues.
    count_zeros_eigvalues : int
        Quantity of zeros eigenvalues.
    algebraic_connectivty : float
        Second eigenvalue of C.

    """
    
    eig_values,eig_vectors=linalg.eig(C)
    abs_eigs=np.abs(eig_values)
    sort_index=np.argsort(abs_eigs)
    #Sort eigenvalues
    eig_values=eig_values[sort_index]
    eig_vectors=eig_vectors[:,sort_index]

    #count zero eigvalues
    count_zeros_eigvalues=len(np.argwhere(np.abs(eig_values)<1e-9))
    #Algebraic connectivity
    algebraic_connectivity=np.abs(eig_values[1])
    
    return eig_values, eig_vectors, count_zeros_eigvalues, algebraic_connectivity
    

    