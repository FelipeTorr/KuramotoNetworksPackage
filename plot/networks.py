#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
#from visbrain.objects import BrainObj, ColorbarObj, SceneObj, SourceObj
import csv

def DrawNetwork(G):
    widths = nx.get_edge_attributes(G, 'weight')
    nodelist = G.nodes()

    plt.figure(figsize=(12,8))

    pos = nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=200, threshold=0.0000001, 
                            weight='weight', scale=1, center=None, dim=2, seed=2)  
    nx.draw_networkx_nodes(G,pos,
                        nodelist=nodelist,
                        node_size=1500,
                        node_color='black',
                        alpha=0.7)
    nx.draw_networkx_edges(G,pos,
                        edgelist = widths.keys(),
                        width=list(widths.values()),
                        edge_color='lightblue',
                        alpha=0.6)
    nx.draw_networkx_labels(G, pos=pos,
                            labels=dict(zip(nodelist,nodelist)),
                            font_color='white')
    plt.box(False)
    # plt.show()

def colorizeMatrix(clusters,C):
    colors=list(clusters.keys())
    
    sorted_indexes=[]
    
    C_colored=np.zeros_like(C)
    for nc,color in enumerate(colors):
        try:   
            for idx in clusters[color]:
                sorted_indexes.append(idx)
                for jdx in range(90):
                    if jdx in clusters[color]:
                        C_colored[idx,jdx]=nc+1
                    else:
                        C_colored[idx,jdx]=len(colors)+1
        except TypeError: #Clusters with only one node
            sorted_indexes.append(int(clusters[color]))
            C_colored[int(clusters[color]),:]=nc+1
    return C_colored, colors, sorted_indexes
    
def colorizeBrain(clusters,C):
    N=90
    x=np.zeros((90,),dtype=float)
    y=np.zeros((90,),dtype=float)
    z=np.zeros((90,),dtype=float)
    s=np.zeros((90,),dtype=float)
    c=np.zeros((90,),dtype=int)
    label=[]
    C_colored, colors, sorted_indexes=colorizeMatrix(clusters, C)
    G=nx.Graph()
    with open('../input_data/Node_AAL90.node', newline='\r\n') as csvfile:
        reader=csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row,n_row in zip(reader,range(N+1)):
            if n_row>0:
                x[n_row-1]=float(row[0])
                y[n_row-1]=float(row[1])
                z[n_row-1]=float(row[2])
                c[n_row-1]=row[3]
                s[n_row-1]=row[4]
                label.append(row[5])
                for n_key, key in enumerate(clusters.keys()):
                    try:
                        if (n_row-1) in clusters[key]:
                            node_color=key
                    except TypeError:
                        if (n_row-1)==clusters[key]:
                            node_color=key
                G.add_node(n_row, pos=(x[n_row-1],y[n_row-1],z[n_row-1]),color=node_color)
    
                for n_col in range(N):
                    G.add_edge(n_row, n_col+1, weight=C[n_row-1,n_col]/np.sum(C))
    pos=nx.get_node_attributes(G,'pos')
    node_xyz=np.array([pos[v] for v in range(1,N+1)])
    nodes,node_color=zip(*nx.get_node_attributes(G, 'color').items())
    return *node_xyz.T, node_color

def plotClustersBrain(clusters,C,threshold_connections=1e-3,figname='clusteredNetwork'):
    from matplotlib.colors import ListedColormap
    N=90
    x=np.zeros((90,),dtype=float)
    y=np.zeros((90,),dtype=float)
    z=np.zeros((90,),dtype=float)
    s=np.zeros((90,),dtype=float)
    c=np.zeros((90,),dtype=int)
    label=[]
    Cn=np.ma.masked_where(C==0, C)
    
    C_colored, colors, sorted_indexes=colorizeMatrix(clusters, C)
    Cn_colored=np.ma.masked_where(C==0, C_colored)
    colors.append('gray')
    clustercolors=ListedColormap(colors)
    G=nx.Graph()
    with open('../input_data/Node_AAL90.node', newline='\r\n') as csvfile:
        reader=csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row,n_row in zip(reader,range(N+1)):
            if n_row>0:
                x[n_row-1]=float(row[0])
                y[n_row-1]=float(row[1])
                z[n_row-1]=float(row[2])
                c[n_row-1]=row[3]
                s[n_row-1]=row[4]
                label.append(row[5])
                for n_key, key in enumerate(clusters.keys()):
                    try:
                        if (n_row-1) in clusters[key]:
                            node_color=key
                    except TypeError:
                        if (n_row-1)==clusters[key]:
                            node_color=key
                G.add_node(n_row, pos=(x[n_row-1],y[n_row-1],z[n_row-1]),color=node_color)
    
                for n_col in range(N):
                    G.add_edge(n_row, n_col+1, weight=C[n_row-1,n_col]/np.sum(C))
            
    fig1=plt.figure()
    ax1 = fig1.add_subplot(2,2,1,projection="3d")
    ax2 = fig1.add_subplot(2,2,2,projection="3d")
    ax3 = fig1.add_subplot(2,2,3)
    ax4 = fig1.add_subplot(2,2,4)
    pos=nx.get_node_attributes(G,'pos')
    node_xyz=np.array([pos[v] for v in range(1,N+1)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])
    nodes,node_color=zip(*nx.get_node_attributes(G, 'color').items())
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
    # nx.draw_networkx(G,pos,ax=ax1,edge_color=weights,edge_cmap=plt.cm.turbo,node_color=node_color,font_color='white')
    ax1.scatter(*node_xyz.T,s=20,c=node_color)
    ax2.scatter(*node_xyz.T,s=20,c=node_color)
    ax1.set_title('Top')
    ax2.set_title('Front')
    # Plot the edges
    # for vizedge,vizweight in zip(edge_xyz,weights):
    #     if vizweight>threshold_connections:
    #         ax1.plot(*vizedge.T, color=plt.cm.jet(vizweight*10*N))
    ax1.set_axis_off()
    ax1.view_init(89,270)
    
    ax2.set_axis_off()
    ax2.view_init(0,90)
    
    im3=ax3.imshow(Cn[sorted_indexes,:][:,sorted_indexes],aspect='equal',cmap=plt.cm.gist_earth_r,interpolation='None')
    im4=ax4.imshow(Cn_colored[sorted_indexes,:][:,sorted_indexes],aspect='equal',cmap=clustercolors,interpolation='None')
    ax3.set_xticks(np.arange(0,91,30))
    ax3.set_yticks(np.arange(0,91,30))
    ax4.set_xticks(np.arange(0,91,30))
    ax4.set_yticks(np.arange(0,91,30))
    
    cb3=fig1.colorbar(im3,ax=ax3)
    cb4=fig1.colorbar(im4,ax=ax4)
    cb3.set_label('weight',fontsize=8)
    cb4.set_label('cluster',fontsize=8)
    cb4.set_ticks(np.arange(1,len(colors)))
    plt.show()
    fig1.savefig(figname+'.pdf',dpi=300)
    return Cn_colored, sorted_indexes
