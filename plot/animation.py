#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from PIL import Image
import os

### Animate Kuramoto model results###
def remove_img():

    [os.remove(file) for file in os.listdir() if file.endswith('.png')]


def Animation(fp_in,name):
    fp_out = name+".gif"

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img, *imgs = [Image.open(f) for f in fp_in]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
                save_all=True, duration=150, loop=0)
    remove_img()


def animateSync(T,dt,Step,act_mat):
        '''
        act_mat is the activity matrix or the phases after integration
        Step is the frame duration in the gif
        '''

        act_mat=act_mat.T
        fp_in = []
        K=1
        times=range(0,int(T/dt), Step)
        cmap = colors.ListedColormap(['k','b','y','g','r'])
        for time in times:
            # print(time)
            f=plt.figure()
            plt.xlim([-1.5, 1.5])
            plt.ylim([-1.5, 1.5])
            plt.plot(np.cos(act_mat[:,time]),np.sin(act_mat[:,time]),'o',markersize=10)
            plt.title('Time='+'%.1f ' % time)
            plt.ylabel(r'$sin(\theta)$')
            plt.xlabel(r'$cos(\theta)$')
            plt.savefig(str(K)+'.png')
            fp_in.append(str(K)+".png")
                
            K=K+1
            plt.close()

            
        Animation(fp_in,"Order Parameter")
        
def animateClusters(T,dt,Step,act_mat,CoherentPhases):
        '''
        act_mat is the activity matrix or the phases after integration
        Step is the frame duration in the gif
        '''

        act_mat=act_mat.T
        fp_in = []
        K=1
        times=range(0,int(T/dt), Step)
        n=CoherentPhases.shape[0]
        cmap = colors.ListedColormap(['k','b','y','g','r'])
        for time in times:
            # print(time)
            f=plt.figure()

            
            for i in range(n-1):
                for j in range(i+1,n):
                    if CoherentPhases[i,j]==1:
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
            plt.close()
        Animation(fp_in,"Order Parameter")