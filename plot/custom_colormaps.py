#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 17:45:42 2023

@author: felipe
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


custom_map = colors.LinearSegmentedColormap.from_list("custom",["gray","yellow","orange","red"])
custom_map1 = colors.LinearSegmentedColormap.from_list("custom",["silver","gold","orange","red"])
custom_map2 = colors.LinearSegmentedColormap.from_list("custom",["gray","goldenrod","orange","pink"])


plt.figure()

x = np.linspace(0.0, 1.0, 100)
rgb=custom_map(x)[:, :3]
rgb1=custom_map1(x)[:, :3]
rgb2=custom_map2(x)[:, :3]
lab = np.sqrt(np.sum(rgb**2,axis=1))
lab1 = np.sqrt(np.sum(rgb1**2,axis=1))
lab2 = np.sqrt(np.sum(rgb2**2,axis=1))

plt.subplot(1,3,1)
plt.scatter(x,lab,c=x,cmap=custom_map)
plt.ylim([0.8,1.8])
plt.subplot(1,3,2)
plt.scatter(x,lab1,c=x,cmap=custom_map1)
plt.ylim([0.8,1.8])
plt.subplot(1,3,3)
plt.scatter(x,lab2,c=x,cmap=custom_map2)
plt.ylim([0.8,1.8])
plt.tight_layout()