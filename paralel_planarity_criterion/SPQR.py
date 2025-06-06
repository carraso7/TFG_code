# -*- coding: utf-8 -*-
"""
Created on Mon May 19 13:02:46 2025

@author: carlo
"""

import networkx as nx
from itertools import combinations


@staticmethod
def get_SPQR_tree(G, TCCs, sep_pairs):  ### TODO REVISAR DOC Y VER QUÉ HACER CON LOS NODOS QUE NO ESTÁN EN NINGUNA COMPONENTE TRICONECTADA.
    """
    Constructs an SPQR-like tree with:
    - Edges between separation pairs (real if in G, virtual otherwise).
    - Edges within TCCs that involve uncategorized nodes.
      (real if in G, virtual otherwise)

    Parameters
    ----------
    G : networkx.Graph
        The original graph.
    TCCs : list of dicts
        Triconnected components with 'node_list' and 'virtual_edges'.
    sep_pairs : list of tuples
        List of separation pairs.

    Returns
    -------
    edges_dict : dict
        {
            'real_edges': list of edges present in G,
            'virtual_edges': list of edges not in G
        }
    SPQR_tree : networkx.Graph
        Graph composed of those edges.
    """
    import networkx as nx

    # Get nodes categorized in any of the triconnected components.
    categorized_nodes = set()
    # print(TCCs) ### TODO PRINT QUITAR
    for tcc in TCCs:
        # print(tcc) ### TODO PRINT QUITAR
        categorized_nodes.update(tcc["node_list"])
    # print(categorized_nodes) ### TODO PRINT QUITAR
    
    
    all_nodes = set(G.nodes())
    uncategorized_nodes = all_nodes - categorized_nodes
    # print("uncategorized: ", uncategorized_nodes) ### TODO PRINT QUITAR
    real_edges = set()
    virtual_edges = set()

    # Process separation pairs
    for u, v in sep_pairs:
        edge = tuple(sorted((u, v)))
        if G.has_edge(*edge):
            real_edges.add(edge)
        else:
            virtual_edges.add(edge)
    
    for e in G.edges():
        if (e[0] in uncategorized_nodes) or (e[1] in uncategorized_nodes):
            real_edges.add(e)
        
    
    # # Process edges inside TCCs involving uncategorized nodes
    # for tcc in TCCs:
    #     tcc_nodes = set(tcc["node_list"])
    #     tcc_uncategorized = tcc_nodes & uncategorized_nodes
    #     print(tcc_uncategorized)
    #     if tcc_uncategorized:
    #         # Add internal real edges
    #         for u in tcc_nodes:
    #             for v in tcc_nodes:
    #                 if u < v and ((u in tcc_uncategorized) or (v in tcc_uncategorized)):
    #                     edge = tuple(sorted((u, v)))
    #                     if G.has_edge(*edge):
    #                         real_edges.add(edge)

    #         # Add virtual edges that involve any uncategorized node and are not real
    #         for u, v in tcc["virtual_edges"]:
    #             edge = tuple(sorted((u, v)))
    #             if (u in tcc_uncategorized or v in tcc_uncategorized) and not G.has_edge(*edge):
    #                 virtual_edges.add(edge)

    # Build SPQR tree
    SPQR_tree = nx.Graph()
    SPQR_tree.add_edges_from(real_edges)
    SPQR_tree.add_edges_from(virtual_edges)

    edges_dict = {
        "real_edges": list(real_edges),
        "virtual_edges": list(virtual_edges),
    }

    return edges_dict, SPQR_tree



@staticmethod
def get_SPQR_tree2(G, TCCs):
    """
    Constructs an SPQR-like tree that includes:
    - all virtual edges from TCCs,
    - real edges between intersecting nodes shared by TCCs,
    - all uncategorized nodes and their incident edges.

    Parameters
    ----------
    G : networkx.Graph
        The original graph.
    TCCs : list of dicts
        Triconnected components, each with 'node_list' and 'virtual_edges'.

    Returns
    -------
    edges_dict : dict
        Contains:
            - 'real_edges': edges between shared nodes that exist in G
            - 'virtual_edges': edges listed as virtual in any TCC
            - 'uncategorized_edges': edges incident to uncategorized nodes
            - 'uncategorized_nodes': nodes not appearing in any TCC
    SPQR_tree : networkx.Graph
        Graph with all edge types and their connected nodes
    """
    import networkx as nx
    from itertools import combinations

    node_to_TCCs = {}
    categorized_nodes = set()

    # Map nodes to the TCCs they appear in
    for idx, tcc in enumerate(TCCs):
        for node in tcc["node_list"]:
            categorized_nodes.add(node)
            node_to_TCCs.setdefault(node, set()).add(idx)

    # --- Shared real edges (intersection nodes that share a real edge) ---
    shared_nodes = [node for node, comps in node_to_TCCs.items() if len(comps) >= 2]
    real_edges = []
    for u, v in combinations(shared_nodes, 2):
        if G.has_edge(u, v):
            real_edges.append(tuple(sorted((u, v))))

    # --- Virtual edges from TCCs ---
    virtual_edges = set()
    for tcc in TCCs:
        for ve in tcc["virtual_edges"]:
            virtual_edges.add(tuple(sorted(ve)))

    # --- Uncategorized nodes and their edges ---
    all_nodes = set(G.nodes())
    uncategorized_nodes = all_nodes - categorized_nodes
    uncategorized_edges = []
    for node in uncategorized_nodes:
        for neighbor in G.neighbors(node):
            edge = tuple(sorted((node, neighbor)))
            uncategorized_edges.append(edge)

    # --- Build SPQR Tree ---
    SPQR_tree = nx.Graph()
    SPQR_tree.add_edges_from(real_edges)
    SPQR_tree.add_edges_from(virtual_edges)
    SPQR_tree.add_edges_from(uncategorized_edges)

    edges_dict = {
        "real_edges": real_edges,
        "virtual_edges": list(virtual_edges),
        "uncategorized_edges": uncategorized_edges,
        "uncategorized_nodes": list(uncategorized_nodes)
    }

    return edges_dict, SPQR_tree


@staticmethod
def get_SPQR_tree1(G, TCCs): #### TODO REVISAR ESTE MÉTODO
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
