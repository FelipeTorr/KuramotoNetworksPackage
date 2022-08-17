#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkx as nx
import matplotlib.pyplot as plt

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