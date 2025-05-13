import networkx as nx
import random
import planarity_criterion
from triconnected_components import TriconnectedFinder

class GraphExamples: ### TODO: HAY QUE PONER LOS IMPORTS ASÍ PERO ES MUY RARO

    @staticmethod
    def get_examples(): ### TODO METER EJEMPLOS APROPIADOS PARA PLANAR PRINTER Y PARA PLANAR CRIT DISTINTOS, CON COMPONENTES TRICO MÁS GRANDES, COMO CON DODECAEDROS O COSAS ASÍ EN EL EXTERIOR
        ### MAKE GRAPH EXAMPLES ###
        graph_examples = []
        
        ## G1 ##
        
        # Graph 1: K4 (Complete graph on 4 vertices) - 1 triply connected component
        G1 = nx.complete_graph(4)
        graph_examples.append(G1)
        
        ## G2 ##
        
        # Graph 2: K5 (Complete graph on 5 vertices) - 1 triply connected component
        G2 = nx.complete_graph(5)
        graph_examples.append(G2)
        
        ## G3 ##
        
        # Graph 3: Two K4s joined by a bridge (2 triply connected components)
        G3 = nx.Graph()
        G3.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)])  # K4-1
        G3.add_edges_from([(4, 5), (5, 6), (6, 7), (7, 4), (4, 6), (5, 7)])  # K4-2
        G3.add_edge(2, 4)  # Single edge bridge
        graph_examples.append(G3)
        
        ## G4 ##
        
        # Graph 4: K4 with an extra node connected to all (1 triply connected component)
        G4 = nx.complete_graph(4)
        G4.add_node(4)
        G4.add_edges_from([(4, 0), (4, 1), (4, 2), (4, 3)])
        graph_examples.append(G4)
        
        ## G5 ##
        
        # Graph 5: A ladder-like graph with extra cross edges (3 triply connected components)
        G5 = nx.Graph()
        edges5 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5),  # Chain
                  (6, 7), (7, 8), (8, 9), (9, 10), (10, 11),  # Parallel Chain
                  (0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11),  # Vertical connections
                  (1, 6), (2, 7), (3, 8), (4, 9), (5, 10)]  # Cross connections
        G5.add_edges_from(edges5)
        graph_examples.append(G5)
        
        ## G6 ##
        
        # Graph 6: A cube graph (4 triply connected components)
        G6 = nx.cubical_graph()  # A cube where opposite vertices connect
        graph_examples.append(G6)
        
        ## G7 ##
        
        # Graph 7: Two K4s joined at two vertices (3 triply connected components)
        G7 = nx.Graph()
        G7.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)])  # K4-1
        G7.add_edges_from([(2, 4), (4, 5), (5, 6), (6, 2), (4, 6), (5, 2)])  # K4-2
        graph_examples.append(G7)
        
        ## G8 ##
        
        # Graph 8: A triangular mesh with increasing connectivity (5 triply connected components)
        G8 = nx.Graph()
        edges8 = [(0, 1), (1, 2), (2, 3), (3, 0),  # Outer cycle
                  (0, 4), (1, 4), (1, 5), (2, 5), (2, 6), (3, 6), (3, 7), (0, 7),  # Inner connections
                  (4, 5), (5, 6), (6, 7), (7, 4)]  # Inner cycle
        G8.add_edges_from(edges8)
        graph_examples.append(G8)
        
        ## G9 ##
        
        # Create a new graph similar to the image
        G9 = nx.Graph()
        
        # Add nodes
        G9.add_nodes_from(range(12))  # Arbitrary node numbers
        
        # Add inner highly connected structure (approximating the central dense part)
        inner_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # K4-like core
            (0, 4), (1, 4), (1, 5), (2, 5), (2, 6), (3, 6), (3, 7), (0, 7),  # Inner connections
            (4, 5), (5, 6), (6, 7), (7, 4)  # Inner cycle
        ]
        
        # Add outer edges extending from the central structure
        outer_edges = [
            (1, 8), (4, 8), (2, 9), (5, 9), (3, 10), (6, 10), (7, 11), (0, 11),
            (8, 9), (9, 10), (10, 11), (11, 8)  # Outer cycle
        ]
        
        # Large external loop (approximating the curve in the image)
        loop_edges = [(8, 12), (12, 9), (12, 10), (12, 11)]  # Adding a large external node
        
        # Add all edges
        G9.add_edges_from(inner_edges + outer_edges + loop_edges)
        
        # Add to graph examples
        graph_examples.append(G9)
        
        
        ### GRAPHS IN THE PAPER ###
        
        ## G10 ##
        
        # Create the graph
        G10_new = nx.Graph()
        
        # Add central nodes
        central_nodes = [0, 1, 2, 3]
        G10_new.add_nodes_from(central_nodes)
        
        # Define additional nodes for K4s (shifted by -1)
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]
        extra_nodes = {pair: (i, i + 1) for i, pair in zip(range(4, 16, 2), pairs)}
        
        # Add edges for each K4 structure
        for (a, b), (x, y) in extra_nodes.items():
            G10_new.add_edges_from([(a, b), (a, x), (a, y), (b, x), (b, y), (x, y)])
        
        # Add to graph examples
        graph_examples.append(G10_new)
        
        ## G11 ##
        
        # Create the graph
        G11 = G10_new.copy()
        G11.add_nodes_from([14, 15])
        G11.add_edges_from([(1, 3), (3, 14), (1, 14), (1, 15), (15, 14)])    
        
        # Add to graph examples
        graph_examples.append(G11)
        
        
        ## G12 ##
        
        # Create the first copy of G10
        G12_part1 = G10_new.copy()
        
        # Relabel nodes in the second copy to avoid conflicts
        mapping = {node: node + 20 for node in G10_new.nodes()}
        G12_part2 = nx.relabel_nodes(G10_new, mapping)
        
        # Create the combined graph
        G12 = nx.Graph()
        G12.add_edges_from(G12_part1.edges())
        G12.add_edges_from(G12_part2.edges())
        
        # Add the common edge between node 1 in each component
        G12.add_edge(1, 21)
        
        # Add to graph examples
        graph_examples.append(G12)

        ### EXAMPLES FROM THE REUNION ###

        # Create two copies of the octahedral graph
        oct1 = nx.octahedral_graph()
        oct2 = oct1.copy()
        
        # Add to graph examples
        graph_examples.append(oct1)
        
        # Relabel nodes in oct2 to avoid conflicts
        offset = len(oct1)
        mapping = {node: node + offset for node in oct2.nodes}
        oct2 = nx.relabel_nodes(oct2, mapping, copy=True)
        
        # Merge both graphs
        combined_graph = nx.compose(oct1, oct2)
        
        # Add two extra edges between the two graphs
        node1_in_oct1 = 0  # Pick a node from oct1
        node2_in_oct1 = 5  # Pick another node from oct1
        node1_in_oct2 = node1_in_oct1 + offset  # Corresponding node in oct2
        node2_in_oct2 = node2_in_oct1 + offset  # Corresponding node in oct2
        
        combined_graph.add_edge(node1_in_oct1, node1_in_oct2)
        combined_graph.add_edge(node2_in_oct1, node2_in_oct2)
        
        contracted = nx.contracted_nodes(combined_graph, 5, 11, self_loops=False)
        contracted = nx.contracted_nodes(contracted, 6, 0, self_loops=False)
        graph_examples.append(contracted)
        
        edges = [
            (0, 2), (0, 3), (0, 4), (0, 5),
            (1, 3), (1, 4), (1, 5),
            (2, 3),
            (3, 4), (4, 5),
            (2, 6), (4, 6),
            (2, 7), (4, 7)
        ]
        
        # Create the graph that fails in triconnected components
        failing_graph = nx.Graph()
        failing_graph.add_edges_from(edges)
        
        graph_examples.append(failing_graph)

        return graph_examples
    
    
    @staticmethod
    def extract_triconnected_subgraph(n=50, seed=42, planar=None):
        """
        Generate random graphs until one contains a triconnected component 
        matching the specified planarity condition.

        Parameters
        ----------
        n : int
            Number of nodes in the random graph.
        seed : int
            Random seed for reproducibility.
        planar : bool or None
            If True, return a planar triconnected component.
            If False, return a non-planar one.
            If None, return any triconnected component.

        Returns
        -------
        G : networkx.Graph
            The full graph containing the desired TCC.
        tcc_subgraph : networkx.Graph
            A subgraph of G that is a matching triconnected component.
        """
        attempts = 0
        while True:
            attempts += 1
            random.seed(seed + attempts)  # Change seed per attempt
            G = nx.gnp_random_graph(n=n, p=0.2, seed=seed + attempts)

            if not nx.is_connected(G):
                G = max((G.subgraph(c) for c in nx.connected_components(G)), key=len).copy()

            finder = TriconnectedFinder()
            TCCs, *_ = finder.triconnected_comps(G)

            if not TCCs:
                continue  # Try another graph if no TCCs found

            # Shuffle TCCs to avoid always picking the same first one
            random.shuffle(TCCs)

            for component in TCCs:
                print(component)
                subG = G.subgraph(component).copy()
                is_planar, _ = nx.check_planarity(subG)

                if planar is None or is_planar == planar:
                    return subG, G
"""
    @staticmethod PREV
    def extract_triconnected_subgraph(n=50, seed=42):
        random.seed(seed)
        G = nx.gnp_random_graph(n=n, p=0.2, seed=seed)
        
        # Make sure it's connected (optional, improves quality)
        if not nx.is_connected(G):
            G = max((G.subgraph(c) for c in nx.connected_components(G)), key=len).copy()
    
        # Use your TriconnectedFinder class
        finder = TriconnectedFinder()
        TCCs, *_ = finder.triconnected_comps(G)
    
        if not TCCs:
            print("No triconnected component found. Try increasing edge density or node count.")
            return G, None
    
        # Select the largest triconnected component
        largest_tcc = max(TCCs, key=len)
        tcc_subgraph = G.subgraph(largest_tcc).copy()
    
        return G, tcc_subgraph
    
"""