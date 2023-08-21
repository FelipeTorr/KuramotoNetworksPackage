#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 23:49:22 2022

@author: felipe
"""

import sys
import os
sys.path.append(os.path.abspath('../'))
import analysis.frequency as frequency
import analysis.synchronization as synchronization

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
import numpy as np
import re
from matplotlib import rc
rc('text', usetex=True)


def handle_close(evt):
        sys.exit()
class VData(object):
    def __init__(self,theta,fs,K):
        self.theta=theta
        self.fs=fs
        self.K=K
        self.kop=synchronization.KuramotoOrderParameter(self.theta)
        self.fig = plt.figure(figsize=(8,6),frameon=False)
        gs=gridspec.GridSpec(3,2,figure=self.fig,width_ratios=[0.3,0.7],height_ratios=[1,1,1],wspace=0.5)
        self.ax1=self.fig.add_subplot(gs[0,0],frame_on=True)
        self.ax2=self.fig.add_subplot(gs[1,0],frame_on=True)
        self.ax3=self.fig.add_subplot(gs[2,0],frame_on=True)
        self.ax4=self.fig.add_subplot(gs[:,1],frame_on=True)
        
        
    def make_frame(self,t,nseg=1):
        #time
        n=int(t*self.fs)
        t_start=t//nseg
        n_second=int(t_start*nseg*self.fs)
        n_now=n-n_second
        tarray=np.linspace(0,nseg,nseg*self.fs+1)        
        
        nseg_kop=10*nseg
        tkop_start=t//nseg_kop
        nkop_second=int(tkop_start*nseg_kop*self.fs)
        nkop_now=n-nkop_second
        tkoparray=np.linspace(0,nseg_kop,nseg_kop*self.fs+1)   
        #Clear axis
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        #Plots
        self.ax1.plot(tarray[0:n_now],np.cos(self.theta[0,n_second:n_second+n_now]),color=plt.cm.tab10(0))
        self.ax1.plot(tarray[n_now-1],np.cos(self.theta[0,n_second+n_now-1]),'o',color=plt.cm.tab10(0)) 
        self.ax1.set_ylabel(r'$\theta_0$',fontsize=12)
        self.ax1.set_ylim([-1.05,1.05])
        self.ax1.set_xlim([0,nseg])
        self.ax1.set_xticks(np.arange(0,nseg,2))
        self.ax1.set_xticklabels(np.arange(t_start*nseg,t_start*nseg+nseg,2))
        
        self.ax2.plot(tarray[0:n_now],np.cos(self.theta[1,n_second:n_second+n_now]),color=plt.cm.tab10(1)) 
        self.ax2.plot(tarray[n_now-1],np.cos(self.theta[1,n_second+n_now-1]),'o',color=plt.cm.tab10(1)) 
        self.ax2.set_ylabel(r'$\theta_1$',fontsize=12)
        self.ax2.set_ylim([-1.05,1.05])
        self.ax2.set_xlim([0,nseg])
        self.ax2.set_xticks(np.arange(0,nseg,2))
        self.ax2.set_xticklabels(np.arange(t_start*nseg,t_start*nseg+nseg,2))
        
        self.ax3.plot(tkoparray[0:nkop_now],self.kop[nkop_second:nkop_second+nkop_now]) 
        self.ax3.set_ylabel(r'$KOP$',fontsize=12)
        self.ax3.set_ylim([-0.1,1.1])
        self.ax3.set_xlim([0,nseg_kop])
        self.ax3.set_xticks(np.arange(0,nseg_kop,2))
        self.ax3.set_xticklabels(np.arange(tkop_start*nseg_kop,tkop_start*nseg_kop+nseg_kop,2))
        
        
        self.ax4.plot(np.cos(self.theta[0,n_second+n_now-10:n_second+n_now]),np.sin(self.theta[0,n_second+n_now-10:n_second+n_now]),':',color=plt.cm.tab10(0))
        self.ax4.plot(np.cos(self.theta[1,n_second+n_now-10:n_second+n_now]),np.sin(self.theta[1,n_second+n_now-10:n_second+n_now]),':',color=plt.cm.tab10(1)) 
        self.ax4.plot(np.cos(self.theta[0,n_second+n_now-1]),np.sin(self.theta[0,n_second+n_now-1]),'o',color=plt.cm.tab10(0))
        self.ax4.plot(np.cos(self.theta[1,n_second+n_now-1]),np.sin(self.theta[1,n_second+n_now-1]),'o',color=plt.cm.tab10(1)) 
        self.ax4.plot([0,self.kop[n_second+n_now-1]],[0,0],'k') 
        self.ax4.set_ylabel('Unit circle',fontsize=12)
        self.ax4.set_ylim([-1.1,1.1])
        self.ax4.set_xlim([-1.1,1.1])
        self.ax4.set_xticks([-1,0,1])
        self.ax4.set_xticklabels([-1,0,1])
        self.ax4.set_yticks([-1,0,1])
        self.ax4.set_yticklabels([-1,0,1])
        self.ax4.set_title('K=%s'%self.K)
        if n==0:
            plt.show()
        return mplfig_to_npimage(self.fig)
    
def main():
    if len(sys.argv) > 0:
        fs=1000
        fps=60
        #Load txt file
        filename = sys.argv[1]
        filenameVideo=re.sub('.npz','Video.mp4',filename)
        tokens=re.split('_',filename)
        for t in tokens:
            if re.search('K',t)!=None:
                K=t[1::]
        print('Loading and plotting '+filename+' ...')
        file=np.load(filename)
        theta=file['theta'][:,0:40000]
        a=VData(theta,fs,K)
        nsamples=np.shape(theta)[1]
        print(nsamples)
        anim = VideoClip(a.make_frame, duration=int((nsamples//fs)))
        anim.write_videofile(filenameVideo, fps=fps)
        a.fig.canvas.mpl_connect('close_event',handle_close)
    else:
        print("Error: filename not valid")
        
if __name__=='__main__':
    main()        