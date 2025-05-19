# -*- coding: utf-8 -*-
"""
Created on Mon May 19 13:02:46 2025

@author: carlo
"""

import networkx as nx
from itertools import combinations

@staticmethod
def get_SPQR_tree(G, TCCs): #### TODO REVISAR ESTE MÉTODO
    """
    Constructs SPQR-like tree edge summary from triconnected components.

    Parameters
    ----------
    G : networkx.Graph
        The original graph.
    TCCs : list of dicts
        Triconnected components as returned by `triconnected_comps`.
        Each dictionary has 'node list' and 'virtual edges'.

    Returns
    -------
    edge_pairs : list of tuples
        List of (u, v) pairs that are in the intersection of components and ARE real edges in G.
    virtual_edge_pairs : list of tuples
        List of (u, v) pairs that are virtual edges (from TCC['virtual edges']).
    connecting_graph : networkx.Graph
        Graph composed of all edge_pairs and virtual_edge_pairs.
    """
    # Track where each node appears
    node_to_TCCs = {}
    for idx, tcc in enumerate(TCCs):
        for node in tcc["node_list"]:
            node_to_TCCs.setdefault(node, set()).add(idx)

    # Find pairs that occur in multiple TCCs (intersection vertices)
    shared_pairs = set()
    for node, tcc_indices in node_to_TCCs.items():
        if len(tcc_indices) >= 2:
            involved_nodes = [n for n in node_to_TCCs if tcc_indices <= node_to_TCCs[n]]
            for u, v in combinations(involved_nodes, 2):
                shared_pairs.add(tuple(sorted((u, v))))

    # Partition into real and virtual edges
    edge_pairs = []
    virtual_edge_pairs = set()
    for tcc in TCCs:
        for ve in tcc["virtual_edges"]:
            virtual_edge_pairs.add(tuple(sorted(ve)))

    real_edges = []
    confirmed_virtual_edges = []

    for u, v in shared_pairs:
        if G.has_edge(u, v):
            real_edges.append((u, v))
        elif (u, v) in virtual_edge_pairs:
            confirmed_virtual_edges.append((u, v))

    # Build the resulting graph
    SPQR_tree = nx.Graph()
    SPQR_tree.add_edges_from(real_edges)
    SPQR_tree.add_edges_from(confirmed_virtual_edges)
    
    edges_dict = {"virtual_edges": confirmed_virtual_edges, 
                  "real_edges": real_edges
                      }
    
    return edges_dict, SPQR_tree
