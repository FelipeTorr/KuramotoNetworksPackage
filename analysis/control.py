#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 14:01:27 2023

@author: felipe
"""

#Canonical Python libraries
import numpy as np
import scipy.signal as signal
from scipy.linalg import schur
#Matplotlib for tests and examples
import matplotlib.pyplot as plt

###############################################################################
### Papers replication
#
def AvgControllability(matrix,c=1):
    w, _ = np.linalg.eig(matrix)
    l = np.abs(w).max()

    # Matrix normalization for discrete-time systems
    A_norm = matrix / (c + l)
    T, U = schur(A_norm, 'real')  # Schur stability
    midMat = np.multiply(U, U).transpose()
    v = np.diag(T)[np.newaxis, :].transpose()
    N = matrix.shape[0]
    P = np.diag(1 - np.matmul(v, v.transpose()))
    P = np.tile(P.reshape([N, 1]), (1, N))
    ac = sum(np.divide(midMat, P))
    return ac


def lowDimensionalCC(matrix,target=None,drivers=None,r_dim=5):
    """
    % Function Header
    % Author: Remy Ben Messaoud
    % Function Name: low_dimensional_control_centrality
    % Associated Work: "Low-dimensional controllability of brain networks"
    % Reference: Messaoud, R. B., Du, V. L., Kaufmann, B. C., Couvy-Duchesne, B., Migliaccio, L., Bartolomeo, P., ... & Fallani, F. D. V. (2023). Low-dimensional controllability of brain networks. arXiv preprint arXiv:2311.11132.
    % Description: Translated to Python 3.8 by Felipe A. Torres

    Parameters
    ----------
    matrix : 2D float Array
        Adjacency matrix of the network
    target : 1D int array
        Vector of indices of nodes composing the target network. Default: The entire network
    drivers : 1D int array
        Vector of indices of nodes to to check as drivers and compute their control centrality.
        Default: The entire network
    r_dim : TYPE
        The low-dimension of the output control, i. e. number of eigenmaps. Default r_dim=5.

    Returns
    -------
    lowCC : 1D float array
        vector of size 1xnDrivers that returns the low-dimensional control centrality for each driver.

    """
    #Sizes and initialization
    n = np.shape(matrix)[0]
    if drivers==None:
        drivers=np.arange(n,dtype=int)
        print('No declared driver indexes. All nodes set as drivers.\n')
    if target==None:
        target=np.arange(n,dtype=int)
        print('No declared target indexes. All nodes set as target.\n')
    if len(target)<2:
        target=np.arange(n,dtype=int)
        print('There must be more than one target. All nodes set as target.\n')
        
    nDrivers = len(drivers)
    targetsize=len(target)
    low_espectralCC = np.zeros((1,nDrivers))
    low_linealCC = np.zeros((1,nDrivers))
    
    #Normalize matrix to A
    lambdaMax=np.sort(np.linalg.eigvals(matrix))[-1]
    A=matrix-1.000001*lambdaMax*np.eye(n)
    # the coef of 1.001 is to ensure Re(Lambda(A))<0 to be stable
    
    
    #Isolate target network
    ## big C matrix for target Control
    bigC=np.zeros((targetsize,n))
    for ktarget in range(targetsize):
        bigC[ktarget,target[ktarget]]=1
    
    #Get the adjacency submatrix of the targets
    targetNet = matrix[np.ix_(target,target)]
    
    # Get the Laplacian eigenmaps of the target network
    targetlaplac = np.diag(np.sum(targetNet,axis=1)) - targetNet
        
    [lambdaNotSorted,VnotSorted] = np.linalg.eig( targetlaplac )
    #lambdaMatNotSorted = np.diag(lambdaNotSorted)
    
    idxAsc =  np.argsort(lambdaNotSorted)
    #lambdaSort= lambdaNotSorted[idxAsc];
    V = VnotSorted[: , idxAsc];
    
    #loop over drivers
    for k in range(nDrivers):
        #Build input matrix B
        driversInd=drivers[k]
        Drivers=np.zeros((1,n))
        Drivers[0,driversInd]=1
        B=np.zeros((n,n))
        #B=B-np.diag(np.diag(B))+np.diag(Drivers[0,:])
        B=np.diag(Drivers[0,:])
        
        #bulid controllability Gramian
        W=np.matmul(np.matmul(A,B),np.matmul(B.T,A.T))
        W=(W+W.T)/2 #Remember the symmetrization of matrix C from AAL90
        lambdaLineal=np.linalg.eigvals(W)
        low_linealCC[0,k]=np.min(np.abs(lambdaLineal))
        
        targetGram=np.matmul(bigC,np.matmul(W,bigC.T))
        
        #Grammian for the eignemodes of the target
        Ceig=V[:,0:r_dim].T 
        Wbar= np.matmul(Ceig,np.matmul(targetGram,Ceig.T))
        targetGramSpec=(Wbar+Wbar.T)/2
        lambdaSpecRed=np.linalg.eigvals(targetGramSpec)
        low_espectralCC[0,k]=np.min(np.abs(lambdaSpecRed))
    
    return low_linealCC,low_espectralCC

###############################################################################
# Build of transfer functions (impulse/step response) 

def build_den_noTransform(z_eigvalues):
    """
    Convolution used for polinomial product
    Prodcut of all roots as [1z,-lambda]
    Return a polinomy of grade r+1, starting at index 0 with highest grade z^r and ending at index r in z^0
    
    Parameters
    ----------
    z_eigvalues : 1D complex array
    Roots of a polinomy. Usually the zeros or the poles of a linear system.
    
    Returns
    -------
    den : 1D float array
        Coeficients of the polinomy, starting for the highest grade, which roots are 'z_eigvalues'.
        [z^N z^{N-1} ... z^{1} 1] 

    """
    
    r=len(z_eigvalues)
    den=np.array([1,-z_eigvalues[0]])
    for m in range(1,r):
        den=np.convolve(den,[1,-z_eigvalues[m]])
    return den

def build_den(eigvalues,dt):
    """
    Convolution used for polinomial product.
    Prodcut of all roots as [1z,-exp(lambda*dt)]
    Return a polinomy of grade r+1, starting at index 0 with highest grade z^r and ending at index r in z^0
    Use this is function if the roots come from the continuous representation to get the discrete representation (casual).  
    
    Parameters
    ----------
    eigvalues : 1D complex array
    Roots of a polinomy. Usually the zeros or poles of a linear system.
    
    dt: float
    Sampling time
    
    Returns
    -------
    den : 1D float array
        Coeficients of the polinomy, starting for the highest grade, which roots are 'exp(eigvalues*dt)'.
        [z^N z^{N-1} ... z^{1} 1] 

    """
    r=len(eigvalues)
    den=np.array([1,-np.exp(eigvalues[0]*dt)])
    for m in range(1,r):
        den=np.convolve(den,[1,-np.exp(eigvalues[m]*dt)])
    return den

def indexes_untilN_remove_m(N,m,zero_index=0):
    """
    This function returns a serie from zero_index until N-1, without m.
    Useful to select all but one element from an array.
    Python indexing: range from 0 to N is the natural numbers' serie from 0 to N-1
    
    Example 1: indexes_untilN_remove_m(N=5,m=2,zero_index=0) returns [0,1,3,4]
    Example 2: indexes_untilN_remove_m(N=6,m=4,zero_index=2) returns [2,3,5]

    Parameters
    ----------
    N : int
        Final value of the serie.
    m : int
        Element to remove
    zero_index : int, optional
        The starting number of the serie. The default is 0.

    Returns
    -------
    indexes : list
        A list with N-zero_index-1 elements starting at 'zero_index', finishing at 'N-1', and without 'm'.

    """
    
    
    indexes=np.arange(zero_index,N)
    indexes=list(indexes)
    indexes.pop(m-zero_index)
    indexes=np.array(indexes)
    return indexes

def build_num(eigvalues,amplitudes,modes,dt):
    """
    Convolution used for polinomial product.
    Prodcut of all roots as b \phi [1z,-exp(lambda*dt)]

    Parameters
    ----------
    eigvalues : 1D complex array
        roots of the numerator polinomy in the form
    amplitudes : 1D float array
        coeficients or weights of each mode
    modes : 2D complex array
        Spatial modes with shape N x r, where N is the number of states or outputs of the system and r is the number of modes.
    dt : float
        Sampling time.

    Returns
    -------
    num : 1D complex array
        numerator polinomy(ies).

    """
    
    #First build the zeros multipliers
    N=np.shape(modes)[0]
    r=len(eigvalues)
    zeros_mult=np.zeros((r,r),dtype=complex) #r is the grade of the polinomy
    num=np.zeros((N,r),dtype=complex)
    for m in range(len(eigvalues)):
        zeros_mult[m,:]=build_den(eigvalues[indexes_untilN_remove_m(r,m)],dt)
    
    #Now multiply (and aggregate) by the amplitude and correspondent mode
    #each row of zeros_mult corresponde to each column of modes, then
    num=modes@np.diag(amplitudes)@zeros_mult
    return num

def canonical_step_numerator(num,den):
    """
    The canonical controllable form of a system with impulse response
    b_0 + \frac{b_1 z^{-1} +b_2 z^{-2}+ ... +b_{r-1}z^{r-1}}{1+a_1 z^{-1} +a_2 z^{-2}+ ... +a_{r-1}z^{r-1}} 
    is the state-space matrix representation x[k+1]=Ax[k]+Bu[k], y[k]=Cx[k]+Du[k] with 
    A=[[0 1 0 ... 0],[0 0 1 ... 0], ...,[0 0 ... 1],[-a_{r-1}  -a_{r-2} ... -a_1]]
    B=[0, 0, 0, 0, ..., b_0]
    C=[ b_{r-1} b_{r-2}  ... b_1]
    D=[b_0]
    
    However, is more usual to have the linear IIR filter representation Y(z)=B'(Z)/A(Z) X(Z), then we need to recalculate the 
    numerator polinomy coefficients to get the canonical B(z).
    
    Parameters
    ----------
    num : 1D complex array
        coefficients of the numerator polinomy
    den : 1D complex array
        coefficients of the denominatro polinomy

    Returns
    -------
    num_canon : 1D complex array
        Canonical numerator [b_1, b_2, ..., b_{r-1}]
    num_ind : complex
        Independent term of the states, or the element of the matrix D (external input weight)
    """
    
    r=len(den)-1
    N=np.shape(num)[0]
    num_canon=np.zeros((N,r))
    num_ind=num[:,0]
    for m in range(1,r):
        #This expression comes form doing partial fractions decomposition
        num_canon[:,m-1]=num[:,m]-num_ind*den[m]
    num_canon[:,r-1]=-num_ind*den[r]
    return num_canon, num_ind

def build_tf_NDMD(eigvalues,modes,Bcontrol,dt):
    """
    Build the diagonal state-space matrices from the DMD eigvalues \Omega, and modes \Phi with sampling time dt
    Parameters
    ----------
    eigvalues : 1D complex array
        eigenvalues or poles of the LTI system
    amplitudes : 1D float array
        amplitudes or weights of each mode
    modes : 2D complex array
        spatial modes from DMD decomposition of shape N x r
    dt : float
        Sampling time

    Returns
    -------
    The linear-time invariant system
    x[k+1]=Ax[k]+Bu[k]
    y[k]=Cx[k]+Du[k]
    
    A : 2D complex array
        Matriz A with shape Nr x Nr.
    B : 2D float array
        Matrix B with shape Nr x N.
    C : 2D complex array
        Matrix C with shape N x Nr.
    D : 2D float array
        Matrix D with shape N x N.
    """
    #Dimensions
    N=np.shape(modes)[0]
    r=len(eigvalues)
    #Numerador and denominador of the transfer function in canonical form
    #State space matrices
    A=np.zeros((N*r,N*r),dtype=complex)
    B=np.zeros((N*r,N))
    C=np.zeros((N,N*r),dtype=complex)
    D=np.zeros((N,N)) #for now is always zero
    
    #Fill the matrices for each node
    for n in range(N):
        A[n*r:(n+1)*r,n*r:(n+1)*r]=np.diag(np.exp(eigvalues*dt))
        if Bcontrol[n,n]!=0:
            B[n*r:(n+1)*r,:]=Bcontrol[n]
        C[n,n*r:(n+1)*r]=modes[n,:]
    return A,B,C,D

def build_tf_canonical(eigvalues,amplitudes,modes,dt):
    """
    Build the canonical controllable state-space matrices from the DMD eigvalues \Omega, ampplitudes b, and modes \Phi with sampling time dt
    Parameters
    ----------
    eigvalues : 1D complex array
        eigenvalues or poles of the LTI system
    amplitudes : 1D float array
        amplitudes or weights of each mode
    modes : 2D complex array
        spatial modes from DMD decomposition of shape N x r
    dt : float
        Sampling time

    Returns
    -------
    The linear-time invariant system
    x[k+1]=Ax[k]+Bu[k]
    y[k]=Cx[k]+Du[k]
    
    A : 2D float array
        Matriz A with shape Nr x Nr.
    B : 2D float array
        Matrix B with shape Nr x N.
    C : 2D float array
        Matrix C with shape N x Nr.
    D : 2D float array
        Matrix D with shape N x N.
    """
    
    #Dimensions
    N=np.shape(modes)[0]
    r=len(eigvalues)
    #Numerador and denominador of the transfer function in canonical form
    num=np.real(build_num(eigvalues,amplitudes,modes,dt))
    den=np.real(build_den(eigvalues,dt))
    num_canon,num_ind=canonical_step_numerator(num,den)
    #State space matrices
    A=np.zeros((N*r,N*r))
    B=np.zeros((N*r,N))
    C=np.zeros((N,N*r))
    D=np.zeros((N,N)) 
    aux_canonical=np.eye(r-1)
    
    #Fill the matrices for each node
    for n in range(N):
        A[n*r:(n+1)*r-1,n*r+1:(n+1)*r]=aux_canonical
        A[(n+1)*r-1,n*r:(n+1)*r]=-np.flip(den[1::])
        B[(n+1)*r-1,n]=1
        C[n,n*r:(n+1)*r]=np.flip(num_canon[n,:])
        D=np.diag(num_ind)
    return A,B,C,D

def build_tf_diagonal(eigvalues,amplitudes,modes,dt):
    """
    Build the diagonal state-space matrices from the DMD eigvalues \Omega, ampplitudes b, and modes \Phi with sampling time dt
    Parameters
    ----------
    eigvalues : 1D complex array
        eigenvalues or poles of the LTI system
    amplitudes : 1D float array
        amplitudes or weights of each mode
    modes : 2D complex array
        spatial modes from DMD decomposition of shape N x r
    dt : float
        Sampling time

    Returns
    -------
    The linear-time invariant system
    x[k+1]=Ax[k]+Bu[k]
    y[k]=Cx[k]+Du[k]
    
    A : 2D complex array
        Matriz A with shape Nr x Nr.
    B : 2D float array
        Matrix B with shape Nr x N.
    C : 2D complex array
        Matrix C with shape N x Nr.
    D : 2D float array
        Matrix D with shape N x N.
    """
    #Dimensions
    N=np.shape(modes)[0]
    r=len(eigvalues)
    #Numerador and denominador of the transfer function in canonical form
    #State space matrices
    A=np.zeros((N*r,N*r),dtype=complex)
    B=np.zeros((N*r,N))
    C=np.zeros((N,N*r),dtype=complex)
    D=np.zeros((N,N)) #for now is always zero
    
    #Fill the matrices for each node
    for n in range(N):
        A[n*r:(n+1)*r,n*r:(n+1)*r]=np.diag(np.exp(eigvalues*dt))
        B[n*r:(n+1)*r,n]=1
        C[n,n*r:(n+1)*r]=modes[n,:]@np.diag(amplitudes)
    return A,B,C,D
        
def build_tf_integrator(eigvalues,amplitudes,modes,dt):
    #TODO: augment the matrix with the integrator tracker dynamics
    #Dimensions
    N=np.shape(modes)[0]
    r=len(eigvalues)
    #Numerador and denominador of the transfer function in canonical form
    num=build_num(eigvalues,amplitudes,modes,dt)
    den=build_den(eigvalues,dt)
    #State space matrices
    A=np.zeros((N*r+N,N*r+N),dtype=complex)
    B=np.zeros((N*r+N,N))
    C=np.zeros((N,N*r+N),dtype=complex)
    # D=np.zeros((N,N)) for now is always zero
    aux_canonical=np.eye(r-1)
    
    #Fill the matrices for each node
    r_tilde=r+1
    for n in range(N):
        A[n*r:(n+1)*r_tilde-1,n*r_tilde+1:(n+1)*r_tilde]=aux_canonical
        A[(n+1)*r_tilde-1,n*r_tilde:(n+1)*r_tilde]=np.fliplr(den[1::])
        B[n*r:(n+1)*r_tilde,n]=1
        C[n,n*r:(n+1)*r_tilde]=np.fliplr(num[n,:])
    return A,B,C

###############################################################################
#Different linear control strategies

def build_R(e_p,N,r,driving_nodes='all'):
    """
    Build the regularization matrix R for u[k]=R x[k] using pole localization. 
    The system representation in the state-space must be the the controllable canonical form. 

    Parameters
    ----------
    e_p : 1D float array
        Difference of desired denominator coeficients - uncontrolled system denominator coeficients.
    N : int
        Number of inputs.
    r : int
        Number of modes.
    driving_nodes : int o 1D int array, optional
        Indices of the input nodes. The default is 'all'.

    Returns
    -------
    R : 2 float array
        Regularization matrix for pole localization.

    """
    
    R=np.zeros((N,N*r))
    if type(driving_nodes)==str:
        if driving_nodes=='all':
            for n in range(N):
                R[n:n+1,r*n:r*(n+1)]=e_p
    elif len(driving_nodes)==1:
        R[driving_nodes:(driving_nodes+1),r*driving_nodes:r*(driving_nodes+1)]=e_p
    else:
        assert len(driving_nodes)<=N, 'The number of indices in driving_nodes must be least or equal to N'
        for targ in driving_nodes:
            R[targ:(targ+1),r*targ:r*(targ+1)]=e_p
    return R

def build_J(N,r):
    """
    J matrix of Ackerman 

    Parameters
    ----------
    N : int
        Observable outputs of the system.
    r : int
        Number of modes/states for each output of the system.

    Returns
    -------
    J : 2D float array
        Matrix of zeros with shape N x Nr, with ones at the positions n, n*r+r-1 for n in the range 0 to N. 
    """
    
    J=np.zeros((N,N*r))
    for n in range(N):
        J[n,n*r+(r-1)]=1
    return J

def build_MC(A,B):
    """
     Controllability matrix

    Parameters
    ----------
    A : 2D float array
        State-space matrix A with shape Nr x Nr.
    B : 2D float array
        State-space matrix B with shape Nr x N. 

    Returns
    -------
    Mc : 2D float array
        Controllability matrix of shape Nr x Nr.
    controllability : float
        Determinant of the matrix Mc.
        The system is controllable if and only if controllability !=0. 

    """
    assert np.shape(A)[0]==np.shape(A)[1],'A must be square'
    assert np.shape(A)[1]==np.shape(B)[0],'A and B must be multipicables'
    #Matriz de controbilidad
    Mc=B
    r=np.shape(B)[0]/np.shape(B)[1]
    for m in range(1,int(r)):    
        Mc=np.hstack((Mc, np.linalg.matrix_power(A,m)@B))
    controllability=np.linalg.matrix_rank(Mc)==r
    return Mc, controllability

def build_MO(A,C):
    """
     Observability matrix

    Parameters
    ----------
    A : 2D float array
        State-space matrix A with shape Nr x Nr.
    C : 2D float array
        State-space matrix B with shape N x Nr. 

    Returns
    -------
    Mo : 2D float array
        Observability matrix of shape Nr x Nr.
    controllability : float
        Determinant of the matrix Mo.
        The system is observable if and only if observability !=0. 

    """
    assert np.shape(A)[0]==np.shape(A)[1],'A must be square'
    assert np.shape(C)[1]==np.shape(A)[0],'C and A must be multipicables'
    #Matriz de controbilidad
    Mo=C
    if np.shape(C)[1]>np.shape(C)[0]:
        r=np.shape(C)[1]/np.shape(C)[0]
    else:
        r=np.shape(A)[0]
    for m in range(1,int(r)):    
        Mo=np.vstack((Mo, C@np.linalg.matrix_power(A,m)))

    observability=np.linalg.matrix_rank(Mo)==r
    return Mo, observability

def build_Characteristic(A,eigvalues,dt):
    """
    Build the characteristic matrix using the coefficients of the characteristic polinomy and the A matrix as independent 'variable'.

    Parameters
    ----------
    A : 2D complex array
        State-space matrix A.
    eigvalues : 1D complex array
        Desired poles of the matrix Phi.
    dt : float
        Sampling time.


    Returns
    -------
    Phi : 2D complex array
        Matrix which eigenvalues correspond to 'eigenvalues' from a linear combination of the matrix A.
    """
    
    characteristic=np.real(build_den(eigvalues, dt))
    r=len(characteristic)
    Phi=characteristic[0]*np.linalg.matrix_power(A, r-1)
    for m in range(1,r):
        Phi+=characteristic[m]*np.linalg.matrix_power(A, r-m-1)
    return Phi
    
def build_K_Ackerman(A,B,eigvalues,dt,r,N=1):
    """
    Return the Ackerman vector of gains. 

    Parameters
    ----------
    A : 2D complex array
        State-space matrix A.
    B : 2D complex array
        State-space matrix B.
    eigvalues : 1D complex array
        Desired poles.
    dt : float 
        Sampling time.
    r : TYPE
        Number of modes/states for each output.
    N : int    
        Number of outputs. The default is 1 as the A, and B matrix are the same for any of the N outputs.
        Also Ackerman gains are defined for systems wiht a single output.

    Returns
    -------
    0 if the system is no controllable
    
    ELSE:
    K: 1D float array
        Vector of gains of u[k]=Kx[k] to achieve the desired poles.
    Phi: 2D complex array 
        The characteristic matrix (the desired state-space matrix A).
    """
    
    assert np.shape(A)[0]==N*r or np.shape(A)[0]==r,'A must be square matrix of size Nr x Nr, or r x r' 
    assert np.shape(B)[0]==N*r or np.shape(B)[0]==r,'B must be a matrix with Nr rows, or r rows' 
    
    J=build_J(1,r)
    Mc, controllability=build_MC(A[0:r,0:r], B[0:r,0:1])
    K=np.zeros((N,N*r))
    if np.abs(controllability)<1e-9:
        print('Not controllable system, Mc is not invertible')
        return K,0
    else:
        Mc_inv=np.linalg.inv(Mc)
        Phi=build_Characteristic(A[0:r,0:r], eigvalues, dt)
        K_single=J@Mc_inv@Phi
       
        for n in range(N):
            K[n:n+1,n*r:(n+1)*r]=K_single
        #For control the input to the system must be replaced by -K@u 
        return K, Phi

def eval_polinomy(polinomy,x):
    """
    Evalute the polinomy dependent in x, P(x)=p_{r}x^{r}+p_{r-1}x^{r-1}+...+p_1x+p_0.
    
    Parameters
    ----------
    polinomy : 1D complex array
        Coefficients of the polinomy.
    x : complex
        Value of x to evalaute P(x).

    Returns
    -------
    value : complex
        Value of P(x).

    """
    value=0
    r=len(polinomy)
    for n in range(r):
        value+=polinomy[n]*x**(r-n-1)
    return value

def c2d(b,a,dt,method='zoh'):
    (num,den,dt)=signal.cont2discrete((b,a), dt=dt,method=method)
    return num, den
    
def response_2ndOrder(Mp,tss,dt):
    """
    Discrete time transfer function for a 2nd order system with overimpulse peak
    amplitude Mp and stabilization time tss. Using dt as sampling time.

    Parameters
    ----------
    Mp : float
        Overimpulse peak amplitude.
    tss : float
        stabilization time.
    dt : float
        sampling time.

    Returns
    -------
    b:  1D float array
        Numerator coeffcients of the system.
    a : 1D float array
        Denominator coeficcient of the system.

    """
    
    zeta=np.log(1/Mp)/np.sqrt(np.pi**2+np.log(1/Mp)**2)
    wn=4/(tss*zeta)
    Gp_b=np.array([wn**2])
    Gp_a=np.array([1, 2*zeta*wn, wn**2])
    b,a=c2d(Gp_b,Gp_a,dt)
    return b[0,:],a
    
def expand_input_matrix(B,SC):
    "scale the inputs of B by the structural connectivity (delays) matrix"
    N=np.shape(SC)[0]
    r=int(np.shape(B)[0]/N)
    outB=np.zeros_like(B)
    for n in range(N):
        #The single input of the node
        input_n=B[n*r:(n+1)*r,n]
        #Copy and scale in each node connected by sc
        col_n=SC[:,n]
        for m,weigth_m in enumerate(col_n):
            outB[m*r:(m+1)*r,n]=input_n*weigth_m
    return outB

def tile_col_matrix(B,N=90):
    "scale the inputs of B by the structural connectivity (delays) matrix"
    r=np.shape(B)[0]
    outB=np.zeros((r,N))
    for n in range(N):
            outB[:,n:n+1]=B/N
    return outB

def expand_output_matrix(C,SC):
    "scale the inputs of C by the structural connectivity (delays) matrix"
    N=np.shape(SC)[0]
    r=int(np.shape(C)[1]/N)
    outC=np.zeros_like(C)
    for n in range(N):
        #The single input of the node
        output_n=C[n,n*r:(n+1)*r]
        #Copy and scale in each node connected by sc
        row_n=SC[n,:]
        for m,weigth_m in enumerate(row_n):
            outC[n,m*r:(m+1)*r]=output_n*weigth_m
    return outC

def simulateLTI(A,B,C,D,dt,Tend,K=None,R=None,S=None,T=None,u=None,r=None,x_init=None):
    Nx=np.shape(A)[0]
    Nu=np.shape(B)[1]
    Ny=np.shape(C)[0]
    time_array=np.arange(0,Tend,dt)
    x=np.zeros((Nx,1))
    y=np.zeros((Ny,len(time_array)))
    
    if u is None:
        u=np.zeros((Nu,len(time_array)))
        u[:,0]=1
    if r is None:
        r=np.ones((Nu,len(time_array)))*dt
    if x_init is None:
        x_init=np.zeros((Nx,1))
    else:
        x[:,0]=x_init[:]
    if K is None:
        K=np.eye(Nu)
    if R is None:
        R=np.eye(Nu)
    if S is None:
        S=np.eye(Ny)
    if T is None:
        T=np.eye(Nu)
    for k,tt in enumerate(time_array):
        #Controller
        yS=S@y[:,k:k+1]
        rT=T@r[:,k:k+1]
        uR=R@K@(rT-yS)
        #System
        try:
            y[:,k:k+1]=np.real(C@x+D@uR)
        except:
            y[:,k]=np.squeeze(C@x+D@uR)
        x=A@x+B@uR
    return y                
    

def build_Hankel(x,q=None):
    N=np.shape(x)[0]
    T=np.shape(x)[1]
    if q is None:
        q=T
    hankel=np.zeros((N,q,q))
    for n in range(N):
        for i in range(q):
            for j in range(q-i):
                hankel[n,i,j]=x[n,i+j]
    return hankel


def RST(desired_eigvalues,Az,Bz,d=0):
    Apz=np.convolve([1,-1],Az)
    nr=len(Az)
    ns=len(Bz)+d-1
    Coefs_matrix=np.zeros((nr,nr))
    # return R,S,T

