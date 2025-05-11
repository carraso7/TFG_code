# -*- coding: utf-8 -*-
"""
Created on Sat May 10 23:37:42 2025

@author: carlo
"""

import numpy as np
from itertools import combinations

import planarity_criterion
import triconnected_components as TCC

import matplotlib.pyplot as plt
import networkx as nx

import numpy as np

def merge_tcc_coordinates1(coord_lists):
    """
    Merge multiple coordinate systems (from TCCs) into a single consistent layout.
    
    Parameters:
    - coord_lists: list of dicts, each mapping node -> (x, y) for a TCC.
    
    Returns:
    - merged_coords: dict mapping node -> (x, y), globally consistent.
    """
    from collections import defaultdict
    
    # Map: node -> list of all coordinates it appears with
    node_positions = defaultdict(list)
    for coord in coord_lists:
        for node, pos in coord.items():
            node_positions[node].append(pos)
    
    merged_coords = {}

    for node, positions in node_positions.items():
        # Average position if node appears in multiple components
        if len(positions) == 1:
            merged_coords[node] = positions[0]
        else:
            avg_x = sum(p[0] for p in positions) / len(positions)
            avg_y = sum(p[1] for p in positions) / len(positions)
            merged_coords[node] = (avg_x, avg_y)

    return merged_coords


def rigid_merge_tcc_coordinates(coord_lists):
    """
    Rigidly merge TCC coordinate dictionaries by aligning shared nodes through translation.
    
    Parameters:
    - coord_lists: list of dicts (node -> (x, y)), one per TCC embedding.
    
    Returns:
    - merged_coords: dict of global positions for all nodes.
    """
    merged_coords = {}             # final result
    merged_nodes = set()           # keep track of what’s already placed

    for i, tcc_coords in enumerate(coord_lists):
        if i == 0:
            # First TCC: add as-is
            merged_coords.update(tcc_coords)
            merged_nodes.update(tcc_coords.keys())
            continue
        
        # Find a common anchor node
        anchor = None
        for node in tcc_coords:
            if node in merged_coords:
                anchor = node
                break
        
        if anchor is None:
            # No common node: could place arbitrarily, or skip
            continue
        
        # Compute translation vector
        x_old, y_old = merged_coords[anchor]
        x_new, y_new = tcc_coords[anchor]
        dx, dy = x_old - x_new, y_old - y_new

        # Translate entire TCC
        for node, (x, y) in tcc_coords.items():
            if node not in merged_coords:
                merged_coords[node] = (x + dx, y + dy)
                merged_nodes.add(node)
            else:
                # If already in, we skip to avoid inconsistency
                # Optionally check consistency here
                continue

    return merged_coords



def draw_graph_with_coordinates(G, coordinates, title="Graph Embedding", show_labels=True, virtual_edges=[]):
    """
    Plots the graph G using the provided coordinates, with optional virtual edges as dashed lines.

    Parameters:
    - G : networkx.Graph
        The input graph.
    - coordinates : dict
        A dictionary mapping each node to a (x, y) coordinate.
    - title : str
        Title for the plot.
    - show_labels : bool
        Whether to show node labels.
    - virtual_edges : list of tuples (optional)
        List of virtual edges (node pairs) to draw as dashed lines.
    """

    import matplotlib.pyplot as plt
    import copy

    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    # Create a shallow copy of G without virtual edges
    G_copy = G.copy()
    for u, v in virtual_edges:
        if G_copy.has_edge(u, v):
            G_copy.remove_edge(u, v)
        elif G_copy.has_edge(v, u):  # in case edge is reversed
            G_copy.remove_edge(v, u)

    # Draw real edges and nodes
    nx.draw_networkx_edges(G_copy, pos=coordinates, edge_color='gray', style='solid', ax=ax)
    nx.draw_networkx_nodes(G_copy, pos=coordinates, node_color='skyblue', node_size=600, ax=ax)
    if show_labels:
        nx.draw_networkx_labels(G_copy, pos=coordinates, font_weight='bold', ax=ax)

    # Draw virtual edges as dashed lines
    for u, v in virtual_edges:
        if u in coordinates and v in coordinates:
            x0, y0 = coordinates[u]
            x1, y1 = coordinates[v]
            ax.plot([x0, x1], [y0, y1], linestyle='--', color='gray', linewidth=1.5)

    plt.title(title)
    plt.axis('equal')
    plt.show()


### TODO QUITAR VERSION ANTIGUA SIN VIRTUAL EDGES
def draw_graph_with_coordinates1(G, coordinates, title="Graph Embedding", show_labels=True):
    """
    Plots the graph G using the provided coordinates.

    Parameters:
    - G : networkx.Graph
        The input graph.
    - coordinates : dict
        A dictionary mapping each node to a (x, y) coordinate.
    - title : str
        Title for the plot.
    - show_labels : bool
        Whether to show node labels.
    """
    plt.figure(figsize=(8, 8))
    nx.draw(
        G,
        pos=coordinates,
        with_labels=show_labels,
        node_color='skyblue',
        edge_color='gray',
        node_size=600,
        font_weight='bold'
    )
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    plt.show()


def get_regular_polygon_positions(peripheral_cycle):
    """
    Maps the nodes in `peripheral_cycle` to points of a regular polygon
    inscribed in the unit circle, in the order provided.

    Returns:
        pos: dict[node] = (x, y)
    """
    import numpy as np
    p = len(peripheral_cycle) - 1 if peripheral_cycle[0] == peripheral_cycle[-1] else len(peripheral_cycle)
    pos = {}
    for i, v in enumerate(peripheral_cycle[:p]):
        angle = 2 * np.pi * i / p
        pos[v] = (np.cos(angle), np.sin(angle))
    return pos


def periph_c_to_embedding(G, peripheral_cycle):
    import matplotlib.pyplot as plt
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve
    
    
    # # Step 1: Map peripheral cycle to convex polygon (unit circle)
    # p = len(peripheral_cycle)
    # pos = {}
    # for i, v in enumerate(peripheral_cycle):
    #     angle = 2 * np.pi * i / p
    #     pos[v] = (np.cos(angle), np.sin(angle))  # convex polygon on unit circle
        
    pos = get_regular_polygon_positions(peripheral_cycle)
    
    # Step 2: Construct the conductance matrix A
    n = len(G.nodes)
    node_list = list(G.nodes)
    node_index = {node: i for i, node in enumerate(node_list)}
    A = lil_matrix((n, n))

    for i, u in enumerate(node_list):
        neighbors = list(G.neighbors(u))
        deg = len(neighbors)
        A[i, i] = deg
        for v in neighbors:
            j = node_index[v]
            A[i, j] = -1

    # Step 3: Build right-hand side and solve the system
    b_x = np.zeros(n)
    b_y = np.zeros(n)

    for v in peripheral_cycle:
        i = node_index[v]
        x, y = pos[v]
        A[i, :] = 0
        A[i, i] = 1
        b_x[i] = x
        b_y[i] = y

    x = spsolve(A.tocsr(), b_x)
    y = spsolve(A.tocsr(), b_y)

    # Step 4: Check for edge crossings
    coordinates = {node_list[i]: (x[i], y[i]) for i in range(n)}
    edges = list(G.edges)
    
    def segments_cross(a, b, c, d):
        def ccw(p, q, r):
            return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])
        return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

    for (u1, v1), (u2, v2) in combinations(edges, 2):
        if len(set([u1, v1, u2, v2])) < 4:
            continue
        if segments_cross(coordinates[u1], coordinates[v1], coordinates[u2], coordinates[v2]):
            return False, coordinates

    return True, coordinates


def get_embbeding(G):
    if (len(G.edges()) > 3 * len(G.nodes()) - 6):
        return False, None ###TODO GESTIONAR RETURNS DE INFO, TODOS LOS DICTS CON LAS MISMAS ENTRADAS
    finder = TCC.TriconnectedFinder()
    TCCs, info = finder.triconnected_comps(G)
    #print("TCCs", TCCs) ### TODO PRINT QUITAR 
    #print(info) ### TODO PRINT QUITAR 
    
    coordinates_list = []
    planar_list = []
    
    criterion = planarity_criterion.PlanarityCriterion()
    
    for tcc_list in TCCs:
        #print("tcc list", tcc_list)
        # Extract the subgraph
        tcc = G.subgraph(tcc_list["node_list"]).copy()
        
        print("nodos subgrafo:", tcc.nodes()) ### TODO PRINT QUITAR 
        print("edges subgrafo:", tcc.edges())  ### TODO PRINT QUITAR 
     
        # Add virtual edges
        tcc.add_edges_from(tcc_list["virtual_edges"])
        
        spanning_tree = criterion.spanning_tree(tcc)
        
        fundamental_cycles = criterion.fundamental_cycles(tcc, spanning_tree)
        
        bridges = criterion.get_bridges(tcc, fundamental_cycles)
        
        truth_assign, info = criterion.get_truth_assigment(
            tcc, fundamental_cycles, bridges
            )
        info["truth_assign"] = truth_assign
        
        if truth_assign is None:
            info["failing tcc"] = tcc
            #print("\n\nCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n\n")
            return False, info
        
        rel_lt, info = criterion.compute_lt(
            tcc, truth_assign, fundamental_cycles, info
            )
        
        peripheral_basis, info = criterion.get_peripheral_basis(
            rel_lt, fundamental_cycles, info
            ) 
        peripheral_cycle = criterion.edges_to_cycle(peripheral_basis[0])
        planar, coordinates = periph_c_to_embedding(tcc, peripheral_cycle)
        draw_graph_with_coordinates(tcc, coordinates, 
                                    virtual_edges=tcc_list["virtual_edges"]
                                    )
        coordinates_list.append(coordinates)
        planar_list.append(planar)
        
    # Combine all TCC coordinates into a global layout
    merged = rigid_merge_tcc_coordinates(coordinates_list) ### TODO TENER EN CUENTA POSIBLES NODOS QUE HAYAN QUEDADO FUERA DE LOS TCCs
    
    # print()
    # print("final graph:")
    # print(merged)
    
    for node in G.nodes: ### TODO VER CUÁL ES LA MEJOR MANERA DE HACER ESTO
        if node not in merged:
            # Find a placed neighbor
            placed_neighbors = [nbr for nbr in G.neighbors(node) if nbr in merged]
            if placed_neighbors:
                ref = placed_neighbors[0]
                x, y = merged[ref]
                merged[node] = (x + 0.5, y + 0.5)  # Simple offset
            else:
                # Completely isolated and unplaced: put at origin or random
                merged[node] = (0.0, 0.0)
                
    # print(merged)
    # # Visualize full graph with global layout
    # draw_graph_with_coordinates(G, merged, title="Merged Global Embedding")
    # print()
    
    return planar_list, coordinates_list


def get_embbeding_not_TCC(G):
    if (len(G.edges()) > 3 * len(G.nodes()) - 6):
        return False, None ###TODO GESTIONAR RETURNS DE INFO, TODOS LOS DICTS CON LAS MISMAS ENTRADAS
    finder = TCC.TriconnectedFinder()
    TCCs, info = finder.triconnected_comps(G)
    #print("TCCs", TCCs) ### TODO PRINT QUITAR 
    #print(info) ### TODO PRINT QUITAR 
    
    criterion = planarity_criterion.PlanarityCriterion()
           
    spanning_tree = criterion.spanning_tree(G)
    
    fundamental_cycles = criterion.fundamental_cycles(G, spanning_tree)
    
    bridges = criterion.get_bridges(G, fundamental_cycles)
    
    truth_assign, info = criterion.get_truth_assigment(
        G, fundamental_cycles, bridges
        )
    info["truth_assign"] = truth_assign
    
    if truth_assign is None:
        return False, []
    
    rel_lt, info = criterion.compute_lt(
        G, truth_assign, fundamental_cycles, info
        )
    
    peripheral_basis, info = criterion.get_peripheral_basis(
        rel_lt, fundamental_cycles, info
        ) 
    peripheral_cycle = criterion.edges_to_cycle(peripheral_basis[0])
    planar, coordinates = periph_c_to_embedding(G, peripheral_cycle)
    
    
    for node in G.nodes: ### TODO VER CUÁL ES LA MEJOR MANERA DE HACER ESTO
        if node not in coordinates :
            # Find a placed neighbor
            placed_neighbors = [nbr for nbr in G.neighbors(node) if nbr in coordinates]
            if placed_neighbors:
                ref = placed_neighbors[0]
                x, y = coordinates[ref]
                coordinates[node] = (x + 0.5, y + 0.5)  # Simple offset
            else:
                # Completely isolated and unplaced: put at origin or random
                coordinates[node] = (0.0, 0.0)
    
    
    draw_graph_with_coordinates(G, coordinates)

        

    return planar, coordinates