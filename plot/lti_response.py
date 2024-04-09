#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:33:29 2024

@author: felipe
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


def plotStep(system,t,ax='None',show=False):
    """
    Plot the step response of an LTI system (discrete time)

    Parameters
    ----------
    system : tuple of LTI system as scipy.signal
        The system could be described by the following tuples:    
        1: (num, den, dt)
        2: (zeros, poles, gain, dt)
        3: (A, B, C, D, dt)
    t : 1D float array
        Array of the time points, preferable with a sampling time equal of the system
    ax : matplotlib.pyplot.axes, optional
        Axis where the plot is shown. The default is 'None' and shows the plot in a new figure.
    show : Boolean, optional
        Shows or not the axes at the end of the plotting function. The default is False.

    Returns
    -------
    ax: The updated matplotlib.pyplot.axes 
    Plot in the indicated 'ax' or in a new matplotlib.pyplot.figure

    """
    
    tt,y=signal.dstep(system,t=t)
    if ax=='None':
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
    if len(y)==1:
        y_plot=y[0]
    else:
        y_plot=np.zeros((len(tt),len(y)))
        for n in range(len(y)):
            y_plot[:,n]=y[n][:,n]
    ax.plot(tt,y_plot,':.')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Step response')
    if show:
        plt.show()
    return ax
        
def plotImpulse(system,t,ax='None',show=False):
    """
    Plot the impulse response of an LTI system (discrete time)

    Parameters
    ----------
    system : tuple of LTI system as scipy.signal
        The system could be described by the following tuples:    
        1: (num, den, dt)
        2: (zeros, poles, gain, dt)
        3: (A, B, C, D, dt)
    t : 1D float array
        Array of the time points, preferable with a sampling time equal of the system
    ax : matplotlib.pyplot.axes, optional
        Axis where the plot is shown. The default is 'None' and shows the plot in a new figure.
    show : Boolean, optional
        Shows or not the axes at the end of the plotting function. The default is False.

    Returns
    -------
    ax: The updated matplotlib.pyplot.axes 
    Plot in the indicated 'ax' or in a new matplotlib.pyplot.figure

    """
    tt,y=signal.dimpulse(system,t=t)
    if ax=='None':
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
    if len(y)==1:
        y_plot=y[0]
    else:
        y_plot=np.zeros((len(tt),len(y)))
        for n in range(len(y)):
            y_plot[:,n]=y[n][:,n]
    ax.plot(tt,y_plot,':.')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Impulse response')
    if show:
        plt.show()
    return ax
        
def plotInput(system,u,t=None,ax='None',show=False):
    """
    Plot the response of an LTI system (discrete time) to the input 'u'

    Parameters
    ----------
    system : tuple of LTI system as scipy.signal
        The system could be described by the following tuples:    
        1: (num, den, dt)
        2: (zeros, poles, gain, dt)
        3: (A, B, C, D, dt)
    u : 2D(1D) float array.
        Array of the inputs of the system. 
        If 2D array, the number of rows is the time lenght and the number of columns must be the same as the system inputs.   
    t : 1D float array, optional
        Array of the time points, preferable with a sampling time equal of the system.
        If None, the time array has the same length than the input array.
    ax : matplotlib.pyplot.axes, optional
        Axis where the plot is shown. The default is 'None' and shows the plot in a new figure.
    show : Boolean, optional
        Shows or not the axes at the end of the plotting function. The default is False.

    Returns
    -------
    ax: The updated matplotlib.pyplot.axes 
    Plot in the indicated 'ax' or in a new matplotlib.pyplot.figure

    """
    if t is not None:
        assert np.shape(u)[0]==np.shape(t)[0], 'The input must be defined for the entire time.'
        try:
            tt,y=signal.dlsim(system,u=u,t=t)
        except:
            tt,y,x=signal.dlsim(system,u=u,t=t)
    else:
        try:
            tt,y=signal.dlsim(system,u=u)
        except:
            tt,y,x=signal.dlsim(system,u=u)
    if ax=='None':
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
    ax.plot(tt,u,':k',label='u(t)')
    ax.plot(tt,y,':.',label='y(t)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Response to u(t)')
    ax.legend()
    if show:
        plt.show()
    return ax