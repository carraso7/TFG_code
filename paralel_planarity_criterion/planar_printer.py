import numpy as np
from itertools import combinations

import planarity_criterion
import triconnected_components as TCC

import matplotlib.pyplot as plt
import networkx as nx

import numpy as np

import os


def draw_graph_with_coordinates(G, coordinates, title="Graph Embedding", 
                                show_labels=True, virtual_edges=[], 
                                save=False, name="triconnected drawing", 
                                dir_name="images"
                                ):
    """
    Plots the graph G using the provided coordinates, with optional virtual
    edges as dashed lines.

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
    
        
    if save:
        os.makedirs(dir_name, exist_ok=True)

        # Set the save path (always PNG)
        print(f'{name}')
        save_path = os.path.join('images', f'{name}.png')
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
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
    """
    Returns a barycentric embedding of triconnected graph G if G is planar

    Parameters
    ----------
    G : NetworkX.graph
        Triconnected graph.
    peripheral_cycle : list of nodes
        Cycle to map to the external face of the embeddding.

    Returns
    -------
    planar
        True if G is planar, false otherwise.
    coordinates
        Coordinates of the barycentric embedding if G is planar.

    """
    import matplotlib.pyplot as plt
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve
        
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
        # Counter clockwise
        def ccw(p, q, r):
            return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])
        return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

    for (u1, v1), (u2, v2) in combinations(edges, 2):
        if len(set([u1, v1, u2, v2])) < 4:
            continue
        if segments_cross(coordinates[u1], coordinates[v1], coordinates[u2], coordinates[v2]):
            return False, coordinates

    return True, coordinates


def get_embbeding(G, save=False, name="triconnected drawing", verbose=1):
    """
    Determines if G is planar and returns an embbeding of each triconnected 
    component of G if it is planar.

    Parameters
    ----------
    G : Networkx.graph
        Graph to get embeddings from the triconnected components.
    save : boolean, optional
        When set to true, save the generated images. The default is False.
    name : str, optional
        Name of the image files if save is set to True. The default is 
        "triconnected drawing".
    verbose : int, optional
        Amount of information printed. The default is 1.

    Returns
    -------
    all(planar_list)
        True if G is planar (because all its triconnected components are 
        planar), False otherwise.
    coordinates_list
        List of coordinates of the embedding of each triconnected component of
        G if G is planar.

    """
    if (len(G.edges()) > 3 * len(G.nodes()) - 6):
        return False, None 
    finder = TCC.TriconnectedFinder()
    TCCs, info = finder.triconnected_comps(G)
    
    coordinates_list = []
    planar_list = []
    
    criterion = planarity_criterion.PlanarityCriterion()
    
    # Make one embedding for each triconnected component
    for i, tcc_list in enumerate(TCCs):
        # Extract the subgraph
        tcc = G.subgraph(tcc_list["node_list"]).copy()
        
        # Add virtual edges
        tcc.add_edges_from(tcc_list["virtual_edges"])
        
        # Perform the first steps of Planar-1 to find a peripheral cycle
        
        spanning_tree = criterion.spanning_tree(tcc)
        
        fundamental_cycles = criterion.fundamental_cycles(tcc, spanning_tree)
        
        bridges = criterion.get_bridges(tcc, fundamental_cycles)
        
        truth_assign, info = criterion.get_truth_assigment(
            tcc, fundamental_cycles, bridges
            )
        info["truth_assign"] = truth_assign
        
        if truth_assign is None:
            info["failing tcc"] = tcc
            return False, info
        
        rel_lt, info = criterion.compute_lt(
            tcc, truth_assign, fundamental_cycles, info
            )
        
        peripheral_basis, info = criterion.get_peripheral_basis(
            rel_lt, fundamental_cycles, info
            ) 
        
        # Get initial peripheral cycle to construct the embedding
        peripheral_cycle = criterion.edges_to_cycle(peripheral_basis[0])
        # Get the embedding coordinates
        planar, coordinates = periph_c_to_embedding(tcc, peripheral_cycle)
        if (verbose >= 0):
            draw_graph_with_coordinates(tcc, coordinates, 
                                        virtual_edges=tcc_list["virtual_edges"],
                                        save=save, name=name + str(i)
                                        )
        coordinates_list.append(coordinates)
        planar_list.append(planar)
        
    # Return true if all the triconnected components are planar
    return all(planar_list), coordinates_list
