#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 00:01:11 2022

@author: felipe
"""

import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.axes as axes

#Polar axis plots
def polar_statistics(angles,axis=0,nan_policy='omit'):
    """
    Polar statisitcs
    Return the mean and standard deviation of circular data.
    np.circmean returns same value 
    np.circstd returns different value
    Parameters
    ----------
    angles : 2D float array
        data of angles.
    axis : int, optional
        Selected axis to calculate the statistic. The default is 0.
    nan_policy : string, optional
        Policy for nan values. The default is 'omit'.

    Returns
    -------
    meanpolar : float
        average angle.
    stdpolar : float
        standard deviation of angles.

    """
        
    sin_angles=np.sin(angles)
    cos_angles=np.cos(angles)
    meanpolar=np.arctan(np.mean(sin_angles,axis=axis)/np.mean(cos_angles,axis=axis))
    stdpolar=np.arctan(np.std(sin_angles,axis=axis)/np.std(cos_angles,axis=axis))
    
    return meanpolar,stdpolar


def circ_hist(a,ax,bins=10,bottom=1,mean_axis=0,density=False,cmap=plt.cm.Blues,colorscale=10.0,plotmeanstd=True):
    """
    Circular histogram with mean value and standard deviation
    
    Parameters
    ----------
    a: 2D floay array 
        angles data
    ax: matplotlib.pyplot.axes
        matplotlib.pyplot.axes with projection='polar'
    bins: int, optional. 
        number of bins in the range [0,360) or [0,2pi). Default value is 10 bins. 
    mean_axis: int, optional. 
        The average is calculated along this axis. Default value is 0 
    density: boolean, optional
        if is True it plots a pmf type histogram. Default value is False 
    cmap: plt.cm.colormap, optional. Default is plt.cm.Blues
        Colormap
    colorscale=float, optional
        Scale factor for the colomap. Default is 10.0
    plotmeanstd: boolean, optional
        Plot or not the mean and standard deviation markers. Default is True
    
    Returns
    -------
    ax: matplotlib.pyplot.axes
        The axes that could be shown in a subplot of a figure.
        Usually the same of the 'ax' parameter.
    """
    
    ###Calcualte the histogram
    hist,bin_edges=np.histogram(a,bins=bins,density=density)
    width = (2*np.pi) / (1.2*bins)
    ax = ax
    ###Calculate the statistics
    mean_theta=stats.circmean(a,axis=mean_axis,nan_policy='omit')
    std_theta=stats.circstd(a,axis=mean_axis,nan_policy='omit')
    ###Create circular bars plot from the histogram data
    bars = ax.bar(bin_edges[0:-1], hist, width=width, bottom=bottom)
    ###Plot the statisitics markers
    if plotmeanstd:
        ax.bar(mean_theta, np.max(hist), width=width/3, bottom=bottom, facecolor='red')
        ax.bar(mean_theta+std_theta, np.max(hist), width=width/4, bottom=bottom, facecolor='black')
        ax.bar(mean_theta-std_theta, np.max(hist), width=width/4, bottom=bottom, facecolor='black')
    ##Change the color of the bars
    # Use custom colors and opacity
    for r, bar in zip(hist, bars):
        bar.set_facecolor(cmap(r / colorscale))
    return ax


