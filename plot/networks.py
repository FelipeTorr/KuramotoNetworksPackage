#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
#from visbrain.objects import BrainObj, ColorbarObj, SceneObj, SourceObj
import csv
import nibabel
from scipy.spatial import KDTree
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pylab import get_cmap

def DrawNetwork(G):
    """
    

    Parameters
    ----------
    G : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
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

def loadNiftyVertices():
    N=90
    Sv=np.zeros((N,3))
    c=np.zeros((N,))
    s=np.zeros((N,))
    labels=[]
    fileNodes='../input_data/Node_AAL90.node'
    try:
        ff=open(fileNodes)
        ff.close()
    except FileNotFoundError:
        fileNodes='../../input_data/Node_AAL90.node'
        ff=open(fileNodes)
        ff.close()
        
    with open(fileNodes, newline='\r\n') as csvfile:
        reader=csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row,n_row in zip(reader,range(N+1)):
            if n_row>0:
                Sv[n_row-1,0]=float(row[0])
                Sv[n_row-1,1]=float(row[1])
                Sv[n_row-1,2]=float(row[2])
                c[n_row-1]=row[3]
                s[n_row-1]=row[4]
                labels.append(row[5])
    return Sv,c,s, labels

def ComputeROIParcels(v,i,vals):
    # v is a long list of nx3 vertices, and i is nx1, where i[k] identifies which
    # 'parcel' v[k] belongs to. Vals is the same length as unique(i) and is the 
    # corrsponding functional value of that parcel
    new = np.zeros([len(i),1])
    for k in range(len(vals)):
        these = i==(k+1)
        new[these] = vals[k]
    return new

def spherefit(X):
    # fit sphere to vertex list and find middle
    
    A = np.array([[np.mean(X[:,0]*(X[:,0]-np.mean(X[:,0]))),
                2*np.mean(X[:,0]*(X[:,1]-np.mean(X[:,1]))),
                2*np.mean(X[:,0]*(X[:,2]-np.mean(X[:,2])))],
                [0,
                np.mean(X[:,1]*(X[:,1]-np.mean(X[:,1]))),
                np.mean(X[:,1]*(X[:,2]-np.mean(X[:,2])))],
                [0,0,
                np.mean(X[:,2]*(X[:,2]-np.mean(X[:,2])))]])
    A = A+A.T
    B = np.array([ [np.mean((np.square(X[:,0])+np.square(X[:,1])+np.square(X[:,2]))*(X[:,0]-np.mean(X[:,0])))],
                   [np.mean((np.square(X[:,0])+np.square(X[:,1])+np.square(X[:,2]))*(X[:,1]-np.mean(X[:,1])))],
                   [np.mean((np.square(X[:,0])+np.square(X[:,1])+np.square(X[:,2]))*(X[:,2]-np.mean(X[:,2])))]])
    Centre = np.linalg.lstsq(A,B,rcond=None) # avoid FutureWarning
    Centre = Centre[0]      # final solution is approx matlab unique solution
    return Centre

def CentreVerts(v):
    nv = v.shape
    v  = v - np.repeat(spherefit(v).T,nv[0],axis=0)
    dv = v - np.repeat(spherefit(v).T,nv[0],axis=0)
    return dv



def alignoverlay_KDtree(mv,sv,o,k=1,scaling_factor=0.1):
    # K-nn closest point search to align / project 90-element overlay (o) 
    # matched to AAL source vertices onto a mesh brain - defined  by the 
    # vertices and faces mv & f.
    
    # Get upper and lower boundaries of overlay
    S  = np.array([o.min(),o.max()])
    
    # read AAL sources - not any more! use supplied source vertices
    #v,l = GetAAL()
    y = mv[:,1].copy()*0
    
    # # centre both vertex lists
    mv = CentreVerts(mv)
    sv  = CentreVerts(sv)
    
    # ensure commmon box boundaries - i.e. we're in the same ballpark
    b = sv.min(axis=0)
    B = sv.max(axis=0)
    
    mv[:,0] = b[0] + [B[0]-b[0]] * (mv[:,0] - mv[:,0].min()) / (mv[:,0].max() - mv[:,0].min() )
    mv[:,1] = b[1] + [B[1]-b[1]] * (mv[:,1] - mv[:,1].min()) / (mv[:,1].max() - mv[:,1].min() )
    mv[:,2] = b[2] + [B[2]-b[2]] * (mv[:,2] - mv[:,2].min()) / (mv[:,2].max() - mv[:,2].min() )
    
    mv = np.around(mv,decimals=2)
    sv  = np.around(sv,decimals=2)
    
    # Build the KD-Tree from the larger mesh
    tree = KDTree(sv)
    #Calculate the radius from the most nearest neighbors
    distances, indices = tree.query(mv,k=k)
    radius=np.percentile(distances,10)
    #find the indices of the k-neighbors of each vertex
    for i in range(len(y)):
        indices=tree.query_ball_point(mv[i,:], r=radius)
        # if 0.8*np.median(distances)<upper_bound:
        for ind in indices:
            
            y[i]=y[i]+scaling_factor*o[ind]
    
    # normalise and rescale
    y = S[0] + [S[1]-S[0]] * (y - y.min()) / (y.max() - y.min() )        
    return distances,y


def loadBrainMesh():
    fileMesh='../input_data/NewSmoothed.gii'
    try:
        ff=open(fileMesh)
        ff.close()
    except:
        fileMesh='../../input_data/NewSmoothed.gii'
    v, f = nibabel.load(fileMesh).get_arrays_from_intent (1008)[0].data, \
                nibabel.load(fileMesh).get_arrays_from_intent (1009)[0].data
    return v, f

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
    N=np.shape(C)[0]
    x=np.zeros((N,),dtype=float)
    y=np.zeros((N,),dtype=float)
    z=np.zeros((N,),dtype=float)
    s=np.zeros((N,),dtype=float)
    c=np.zeros((N,),dtype=int)
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
                print(node_color)
                G.add_node(n_row, pos=(x[n_row-1],y[n_row-1],z[n_row-1]),color=node_color)
    
                for n_col in range(N):
                    G.add_edge(n_row, n_col+1, weight=C[n_row-1,n_col]/np.sum(C))
    pos=nx.get_node_attributes(G,'pos')
    node_xyz=np.array([pos[v] for v in range(1,N+1)])
    nodes,node_color=zip(*nx.get_node_attributes(G, 'color').items())
    return *node_xyz.T, node_color

def plotClustersBrain(clusters,C,threshold_connections=1e-3,figname='clusteredNetwork'):
    from matplotlib.colors import ListedColormap
    N=np.shape(C)[0]
    x=np.zeros((N,),dtype=float)
    y=np.zeros((N,),dtype=float)
    z=np.zeros((N,),dtype=float)
    s=np.zeros((N,),dtype=float)
    c=np.zeros((N,),dtype=int)
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

def plotSubnetworks(subnetworks,color='blue',N=90,figname='subnetworks',non_color='gray',dt=1e-3):
    """
    Plot the top and front vision of the 90 nodes in the brain positions.
    Colorize the nodes following the key colors in the dictionary *subnetworks*.

    Parameters
    ----------
    subnetworks : dict
        A dicitionary which keys are color names and their values are a list of 
        nodes that correspond to a particular subnetwork.
    N : int, optional
        Number of nodes. For further development with more atlas, for now it is always the default: 90.
    figname : str, optional
        Name of the generate and stored figure. The default is 'subnetworks'.

    Returns
    -------
    axes. List of matplotlib.axes with the top[0] and front[1] perspectives. 
    
    Also Saves a .png file in the disk.

    """
    M=np.shape(subnetworks)[0]
    rows=int(M//4+1)        
    
    fig1=plt.figure()
    from matplotlib.colors import ListedColormap
    for n,subnet in enumerate(subnetworks):
        textTimes='start: %.2f s duration: %.2f s'%(subnet[0]*dt,subnet[1]*dt)
        subnetwork={}
        subnetwork[color]=subnet[2][0]
        other_nodes=[node for node in range(N) if node not in subnet[2][0]]
        subnetwork['gray']=other_nodes
        x=np.zeros((N,),dtype=float)
        y=np.zeros((N,),dtype=float)
        z=np.zeros((N,),dtype=float)
        s=np.zeros((N,),dtype=float)
        c=np.zeros((N,),dtype=int)
        label=[]
        C=np.ones((N,N))
        colors_subnets=list(subnetwork.keys())
        colors_subnets.append(non_color)
        clustercolors=ListedColormap(colors_subnets)
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
                    for n_key, key in enumerate(subnetwork.keys()):
                        try:
                            if (n_row-1) in subnetwork[key]:
                                node_color=key
                        except TypeError:
                            if (n_row-1)==subnetwork[key]:
                                node_color=key
                        
                                
                    G.add_node(n_row, pos=(x[n_row-1],y[n_row-1],z[n_row-1]),color=node_color)
        
                    for n_col in range(N):
                        G.add_edge(n_row, n_col+1, weight=C[n_row-1,n_col]/np.sum(C))
    
        ax1 = fig1.add_subplot(rows,4,n+1,projection="3d")
        
        pos=nx.get_node_attributes(G,'pos')
        node_xyz=np.array([pos[v] for v in range(1,N+1)])
        edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])
        nodes,node_color=zip(*nx.get_node_attributes(G, 'color').items())
        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
        # nx.draw_networkx(G,pos,ax=ax1,edge_color=weights,edge_cmap=plt.cm.turbo,node_color=node_color,font_color='white')
        ax1.scatter(*node_xyz.T,s=2,c=node_color)
        #ax1.set_title(textTimes,fontsize=8)
        ax1.set_axis_off()
        ax1.view_init(90,270)
    fig1.tight_layout()
    fig1.savefig(figname+'.pdf',dpi=300)
    

def plotAAL90Brain(data90,k=3,interpolation='max',orientation=[90,90],alpha=0.6,cmap_name='turbo',cmap_nodes='turbo',ax='None',plot_nodes=False,show_plot=False):
    #Load matched mesh to the AAL116 regions
    AALv,s,c,labels=loadNiftyVertices()
    #Load the smoothed mesh with lower quantiy of vertices
    v,f= loadBrainMesh()
   
    #Translate the 90 data points to the AAL90 mesh
    # overlay = ComputeROIParcels(AALv,AALi,data90)
    #Translate the AAL90 mesh to the smoothed mesh
    distances,y = alignoverlay_KDtree(v,AALv,data90,scaling_factor=1,k=k) 
    
    #Define the face color as the mean/median/other of the vertices
    colors_indx=np.zeros((np.shape(f)[0],))
    for n in range(np.shape(f)[0]):
        if interpolation=='none':
            if np.count_nonzero(y[f[n,:]])>0:
                colors_indx[n]=np.max(data90)
        else:
            if interpolation=='mean':
                val=np.mean(y[f[n,:]])
            elif interpolation=='median':
                val=np.median(y[f[n,:]])
            elif interpolation=='min':
                val=np.min(y[f[n,:]])
            else:
                val=np.max(y[f[n,:]])
            colors_indx[n]=val
            #Normalice the colors in the range [0,1]
            colors_indx=(colors_indx-np.min(colors_indx))/(np.max(colors_indx)-np.min(colors_indx)) 
    
    #Plot
    normalized_data90=(data90-np.min(data90))/(np.max(data90)-np.min(data90))
    if cmap_name=='custom_plus_black':
        import matplotlib.colors as colors
        cmap = colors.LinearSegmentedColormap.from_list("custom",["silver","gold","orange","red","black"])
    elif cmap_name=='custom_map':
        import matplotlib.colors as colors
        cmap = colors.LinearSegmentedColormap.from_list("custom",["silver","gold","orange","red"])
    else:
        cmap = get_cmap(cmap_name)
    cmap_nodes = get_cmap(cmap_nodes)
    if ax=='None':
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    v1=0.01*v[f[::,:]]
    collection = Poly3DCollection(v1,edgecolors=None) 
    p3dc=ax.add_collection(collection)
    colors_alpha=cmap(colors_indx)[::]
    # p3dc.set(alpha=alpha)
    if plot_nodes:
        p3dc.set_fc(colors_alpha*0)
        for j in range(90):
            if data90[j]!=0:
                ax.plot([0.01*AALv[j,0],0.01*AALv[j,0]],[0.01*AALv[j,1],0.01*AALv[j,1]],[0.01*AALv[j,2],0.01*AALv[j,2]],'o',color=cmap_nodes(normalized_data90[j]),markersize=15)
    else:
        colors_alpha[:,3]=alpha+(colors_indx>0)*(1-alpha)
        p3dc.set_fc(colors_alpha)
    ax.set_ylim([-0.7,0.45])
    ax.set_xlim([-0.5,0.5])
    if orientation[0]==270:
        ax.set_ylim([-0.85,0.38])
        ax.set_xlim([-0.5,0.5])
    ax.set_axis_off()
    ax.view_init(orientation[0],orientation[1])
    if show_plot:
        plt.show()
    return ax

def plotAAL90FC(FC,data90='ones',interpolation='max',orientation=[90,90],cmap_name='turbo',ax='None',show_plot=False):
    N=90
    k=3
    indexes=np.triu_indices(90,k=1)
    
    if data90=='ones':
        data90=np.ones((N,))
    elif data90=='intensity':
        data90=np.sum(FC,axis=0)
    elif data90=='degree':
        binaryFC=np.zeros_like(FC)
        binaryFC[FC!=0]=1
        data90=np.sum(binaryFC,axis=0)
    elif data90=='binary':
        data90=np.zeros((N,))
        binaryFC=np.zeros_like(FC)
        binaryFC[FC!=0]=1
        data90[np.sum(binaryFC,axis=0)>0]=1
    #Load matched mesh to the AAL116 regions
    AALv,s,c,labels=loadNiftyVertices()
    #Load the smoothed mesh with lower quantiy of vertices
    v,f= loadBrainMesh()
   
    #Translate the 90 data points to the AAL90 mesh
    # overlay = ComputeROIParcels(AALv,AALi,data90)
    #Translate the AAL90 mesh to the smoothed mesh
    distances,y = alignoverlay_KDtree(v,AALv,data90,scaling_factor=1,k=k) 
    
    #Define the face color as the mean/median of the vertices
    min_y=np.min(y)
    max_y=np.max(y)
    colors_indx=np.zeros((np.shape(f)[0],))
    for n in range(np.shape(f)[0]):
        if interpolation=='none':
            if np.count_nonzero(y[f[n,:]])>0:
                colors_indx[n]=1.0
        if interpolation=='mean':
            val=np.mean(y[f[n,:]])
        elif interpolation=='median':
            val=np.median(y[f[n,:]])
        elif interpolation=='min':
            val=np.min(y[f[n,:]])
        else:
            val=np.max(y[f[n,:]])
        colors_indx[n]=(val-min_y)/(max_y-min_y)
     
    #Plot
    cmap = get_cmap(cmap_name)
    if ax=='None':
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    v1=0.01*v[f[::,:]]
    collection = Poly3DCollection(v1,edgecolors=None) 
    p3dc=ax.add_collection(collection)
    p3dc.set_fc(cmap(colors_indx)[::])
    p3dc.set(alpha=0.1)
    ax.set_xlim([-0.5,0.5])
    if orientation[0]==270:
        ax.set_ylim([-0.3,0.3])
        ax.set_xlim([-0.5,0.5])
    else:
        ax.set_ylim([-0.7,0.7])
    
    ax.set_axis_off()
    for i,j in zip(indexes[0],indexes[1]):
        if FC[i,j]!=0:
            ax.plot([0.01*AALv[i,0],0.01*AALv[j,0]],[0.01*AALv[i,1],0.01*AALv[j,1]],[0.01*AALv[i,2],0.01*AALv[j,2]],color=plt.cm.RdBu_r(FC[i,j]),linewidth=1.2*FC[i,j])
            ax.plot([0.01*AALv[j,0],0.01*AALv[j,0]],[0.01*AALv[j,1],0.01*AALv[j,1]],[0.01*AALv[j,2],0.01*AALv[j,2]],'o',color='cyan',markersize=1)
    ax.view_init(orientation[0],orientation[1])
    if show_plot:
        plt.show()
    return ax
    



    
