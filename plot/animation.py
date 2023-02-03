#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from PIL import Image
import os

### Animate Kuramoto model results###
def remove_img(fp_in):
    """
    Remove all .png files inside the fp_in list
    
    Use with caution.
    
    Parameters
    ----------
    fp_in : list of filenames
        List with the frame images.
        
    Returns
    -------
    None.

    """
    
    [os.remove(file) for file in fp_in if file.endswith('.png')]


def Animation(fp_in,name):
    """
    Makes a gift with the images inside the fp_in list.

    Parameters
    ----------
    fp_in : list of filenames
        List with the frame images.
    name : str
        Output name of the GIF file. The extension is not required, the internal code assigns the .gif extension.

    Returns
    -------
    None.
        But it saves the GIF file as ' **name** .gif '  in the hard disk. 
   
    """
    
    fp_out = name+".gif"

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img, *imgs = [Image.open(f) for f in fp_in]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
                save_all=True, duration=150, loop=0)
    remove_img(fp_in)


def animateSync(T,dt,Step,act_mat,which_nodes=[0]):
    """
    Makes a gift with the activity of the nodes specified by wich_nodes plotted in the unitary circle.

    Parameters
    ----------
    T : float
        Total simulation time.
    dt : float
        Integration time step.
    Step :float
        The frame duration in the gif
    act_mat : TYPE
        The activity matrix or the phases after integration. Size N x T.

    Returns
    -------
    None.
        But it saves 'CircularSynchronization.gif' in the hard disk.
    """
        
    act_mat=act_mat.T
    fp_in = []
    K=1
    times=range(0,int(T/dt), Step)
    for time in times:
        # print(time)
        plt.figure()
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.plot(np.cos(act_mat[which_nodes,time]),np.sin(act_mat[which_nodes,time]),'o',markersize=10)
        plt.title('Time='+'%.1f ' % time)
        plt.ylabel(r'$sin(\theta)$')
        plt.xlabel(r'$cos(\theta)$')
        plt.savefig(str(K)+'.png')
        fp_in.append(str(K)+".png")
    
        K=K+1
        plt.clf()

    plt.close()
    Animation(fp_in,"CircularSynchronization")
        
def animateClusters(T,dt,Step,act_mat,CoherentPhases,FC_threshold):
    """
    Makes a gift with the activity of the nodes specified by wich_nodes plotted in the unitary circle.
    But only if the FC gives that the phases between the nodes are coherent (>=FC_threshold).

    Parameters
    ----------
    T : float
        Total simulation time.
    dt : float
        Integration time step.
    Step :float
        The frame duration in the gif
    act_mat : float 2D array
        The activity matrix or the phases after integration. Size N x T.

    CoherentPhases :float 2D array
        The FC matrix of act_mat. Size N x N.
    
    Returns
    -------
    None.
        But it saves 'CoherentNodes.gif' in the hard disk.
    """

    act_mat=act_mat.T
    fp_in = []
    K=1
    times=range(0,int(T/dt), Step)
    n=CoherentPhases.shape[0]
    for time in times:
        # print(time)
        plt.figure()        
        for i in range(n-1):
            for j in range(i+1,n):
                if CoherentPhases[i,j]>=FC_threshold:
                    plt.xlim([-1.5, 1.5])
                    plt.ylim([-1.5, 1.5])
                    Seq=[[i],[j]]
                    plt.plot(np.cos(act_mat[Seq,time]),np.sin(act_mat[Seq,time]),'o',markersize=10)
                    # plt.plot(,'o',markersize=10)
        plt.title('Time='+'%.1f ' % time)
        plt.ylabel(r'$sin(\theta)$')
        plt.xlabel(r'$cos(\theta)$')
        plt.savefig(str(K)+'.png')
        fp_in.append(str(K)+".png")
            
        K=K+1
        plt.clf()
    plt.close()
    Animation(fp_in,"CoherentNodes")