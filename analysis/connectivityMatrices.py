#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### Connectivity matrix utilities
import numpy as np
import numpy.linalg as linalg
from scipy.io import loadmat

def loadConnectome(No_nodes):
    """
    

    Parameters
    ----------
    No_nodes : int
        Number of nodes to load.

    Returns
    -------
    C : 2D float array
        Conectome matrix.

    """
    C = loadmat('../input_data/AAL_matrices.mat')['C']
    if No_nodes>90:
        print('Max No. Nodes is 90')
        No_nodes=90

    n = No_nodes
    C=C[:n,:n]
    C[np.diag(np.ones(n))==0] /= C[np.diag(np.ones(n))==0].mean()
    
    return C

def loadDelays(No_nodes):
    """
    Load the matrix of delays between nodes
    Parameters
    ----------
    No_nodes : int
        Number of nodes to load.

    Returns
    -------
    D : 2D float array
        Matrix of delays, unit: seconds.

    """
    
    D = loadmat('../input_data/AAL_matrices.mat')['D']
    D /= 1000 # Distance matrix in meters
    
    return D

def loadLabels():
    file=loadmat('../input_data/AAL_labels.mat')
    labels=file['label90']
    return labels


def DegreeMatrix(C):
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

def AdjacencyMatrix(C):
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

def Intensities(C):
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

def BooleanDegree(C):
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
    
    A=AdjacencyMatrix(C)
    k_i=np.sum(A,axis=1)
    return k_i

def Laplacian(C):
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
    
    D=DegreeMatrix(C)
    L=D-C
    return L

def eigen(C):
    """
    Parameters
    ----------
    C : 2D float square array
        Connectivity matrix.

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
    algebraic_connectivty=np.abs(eig_values[1])
    
    return eig_values, eig_vectors, count_zeros_eigvalues, algebraic_connectivty
    