#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import metis
from networkx.algorithms.community import k_clique_communities

def Clustering(G,No_Clusters):
    color_map = []
    if No_Clusters>10:
        print("Please Edit the code to account for more than 10 clusters")
        No_Clusters=10
    # No_Clusters=9 # Can be assigned, But add extra colors below
    (edgecuts, parts) = metis.part_graph(G, No_Clusters,recursive=True)
    colors = ['red','blue','green','brown','yellow','black','magenta','olive','cyan','purple']
    for i, p in enumerate(parts):
        G.nodes[i]['color'] = colors[p]
        color_map.append(colors[p])
    return(G,color_map)