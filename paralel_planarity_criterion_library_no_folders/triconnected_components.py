# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 14:33:45 2025

@author: carlo
"""
### TODO VER SI SOBRA ALGÚN IMPORT
import networkx as nx
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations

class TriconnectedFinder():
    
    def __find_sep_pairs(self, G):
        """
        Finds and returns all separation pairs of graph G
        
        Parameters
        ----------
        G : networkx.Graph 
            Graph to find separation pairs.

        Returns
        -------
        sep_pairs : list
            List of tuples. Each tuple represents a separation pair of `G`.

        """
        sep_pairs = []
        for sp in list(combinations(G.nodes, 2)): # paralelizable
            H = G.copy()
            H.remove_nodes_from(sp)
            if not nx.is_connected(H):
                sep_pairs.append(sp)
        return sep_pairs
    
    def __find_connected_components(self, G, sep_pairs):
        """
        For each separation pair, labels all the nodes depending
        on the component they belong when deleting the separation pair

        Parameters
        ----------
        G : networkx.Graph 
            Graph to label connected components.
        sep_pairs : list
            List of tuples. Each tuple represents a separation pair of `G`.

        Returns
        -------
        connected_components : dict
            Dictionary with one key for each separation pair in `sep_pairs`.        
            The content of each key is a dict with one key for each node of `G` 
            not in the separation pair. The content of each node is the 
            component it belongs to in the graph ignoring the separation pair. 

        """
        # Hacer que sea un diccionario con claves cada pareja
        connected_components = {}
        for sep_pair in sep_pairs:
            H = G.copy()
            H.remove_nodes_from(sep_pair)
            
            # Assign labels to nodes based on their connected component
            component_labels = {node: i for i, component in enumerate(nx.connected_components(H)) for node in component}
            connected_components[sep_pair] = component_labels
        return connected_components
    
    def __find_relation_R(self, G, connected_components):
        """
        Finds the relation R described in the paper for each pair of nodes in 
        graph `G`. ### TODO INCLUIR REFERENCIA PAPER

        Parameters
        ----------
        G : networkx.Graph
            Graph to find relation R.
        connected_components : dict
            Dictionary representing the connected components of the graph 
            deleting each separation pair (described in 
            `__find_connected_components`).

        Returns
        -------
        relation_R : list
            List of size `G.nodes()` * `G.nodes()`. Nodes with indexes i and j
            are R-related if and only if `relation_R[i][j]` is true.

        """
        ### TODO hacer que el index se calcule una sola vez para todos los métodos que lo usan
        ### TODO ESTO SE PUEDE HACER CON UNA MATRIZ TRIANGULAR O SPARSE MEJOR O INCLUSO CON ALGUNA CLASE DE RELACIONES
        node_index = {node: i for i, node in enumerate(G.nodes)}
        relation_R = [[True for _ in range(len(G.nodes))] for _ in range(len(G.nodes))]
        for pair in list(combinations(G.nodes, 2)):
            ## TODO NO HACE FALTA ESTE BOOLEANO
            related = True
            for sep_pair in connected_components.keys():
                if (pair[0] not in sep_pair) and (pair[1] not in sep_pair):
                    if connected_components[sep_pair][pair[0]] != connected_components[sep_pair][pair[1]]:
                        # print("sep_pair", sep_pair)
                        # print("pair", pair)
                        related = False
                        #break ## TODO CHECKEAR ESTE BREAK
            relation_R[node_index[pair[0]]][node_index[pair[1]]] = related
            relation_R[node_index[pair[1]]][node_index[pair[0]]] = related
                
        return relation_R
    
    
    def __find_relation_T(self, G, relation_R):
        """
        Find relation T described in the paper for graph G.  ### TODO escribir referencia paper

        Parameters
        ----------
        G : networkx.Graph
            Graph to find relation T.
        relation_R : list
            List representing relation R of G as described in 
            `__find_relation_R`.

        Returns
        -------
        relation_T : list
            List of tuples representing relation T. Three nodes are related if 
            and only if they are in a tuple in the list `relation_T`.

        """
        # TODO SACAR EL NODE_INDEX UNA SOLA VEZ Y POASARLO A LOS 3 MÉTODOS QUE SE NECESITAN
        node_index = {node: i for i, node in enumerate(G.nodes)}
        relation_T = []
        for trio in list(combinations(G.nodes, 3)):
            if relation_R[node_index[trio[0]]][node_index[trio[1]]] and relation_R[node_index[trio[0]]][node_index[trio[2]]] and relation_R[node_index[trio[2]]][node_index[trio[1]]]:
                relation_T.append(trio)
        return relation_T
    
    def __find_triply_connected_from_T_R(self, G, rel_T, rel_R):
        """
        Function to find triply connected components from relations T and R of 
        graph ´G´

        Parameters
        ----------
        G : networkx.Graph
            Graph to find triconnected components.
        rel_T : lsit
            Relation T of ´G´ as described in ´__find_relation_T´.
        rel_R : list
            Relation R of ´G´ as described in ´__find_relation_R´.

        Returns
        -------
        list
            List of frozensets. Each set represents the nodes of a triconnected
            component of ´G´.

        """
        node_index = {node: i for i, node in enumerate(G.nodes)}
        triply_components = []
        for rel_T_elem in rel_T:
            triply_component = list(rel_T_elem)
            for node in G.nodes:
                if rel_R[node_index[rel_T_elem[0]]][node_index[node]] and rel_R[node_index[rel_T_elem[1]]][node_index[node]] and rel_R[node_index[rel_T_elem[2]]][node_index[node]]:
                    triply_component.append(node)
            triply_components.append(triply_component)
        return list(set(frozenset(comp) for comp in triply_components))
    
    def find_triply_connected_comps(self, G):
        """
        TODO: - RENOMBRARR COMO BICONNECTED COMPS Y CAMBIAR EN EL RESTO Y CAMBIAR RETURN CON INFO
              - CREAR UN MÉTODO QUE TAMBIÉN DEVUELVA LA INFO Y OTRO QUE NO???
        Parameters
        ----------
        G : networkx.Graph
            Graph to find triconnected components.

        Returns
        -------
        TCCs : list
            List of frozensets. Each set represents the nodes of a triconnected
            component of ´G´.
        all_relation_T : TYPE
            DESCRIPTION.
        all_relation_R : TYPE
            DESCRIPTION.
        all_connected_components : TYPE
            DESCRIPTION.
        all_sep_pairs : TYPE
            DESCRIPTION.

        """
        TCCs = []
        all_relation_T = []
        all_relation_R = []
        all_connected_components = []
        all_sep_pairs = []
        
        # Find biconnected components and create subgraphs
        bicomponents = list(nx.biconnected_components(G))
        
        # subgraphs = [G.subgraph(component).copy() for component in bicomponents]
        subgraphs = [G.subgraph(component).copy() for component in bicomponents if len(component) >= 3]
        
        for subgraph in subgraphs:
            sep_pairs = self.__find_sep_pairs(subgraph)
            connected_components = self.__find_connected_components(subgraph, sep_pairs)
            relation_R = self.__find_relation_R(subgraph, connected_components)
            relation_T = self.__find_relation_T(subgraph, relation_R)
            TCC = self.__find_triply_connected_from_T_R(subgraph, relation_T, relation_R)
            
            TCCs.extend(TCC)
            all_relation_T.append(relation_T)
            all_relation_R.append(relation_R)
            all_connected_components.append(connected_components)
            all_sep_pairs.append(sep_pairs)
        
        return TCCs, all_relation_T, all_relation_R, all_connected_components, all_sep_pairs



import networkx as nx
import matplotlib.pyplot as plt
import textwrap  # For wrapping long lines

class ConnectedComponentsDrawer(): ### TODO LEER SI EN PYTHON ES CORRECTO PONER MÁS DE UNA CLASE EN EL MISMO ARCHIVO O DEBEN ESTAR EN ARCHIVOS SEPARADOS. 
### TODO ESCRIBIR A NETWORKX PARA VER SI QUIEREN MI LIBRERÍA

    def print_n_connected_components(self, G, NCC, max_line_length=40, N=-1):
        # Initialize classification dictionaries
        node_classes = {}  # Store which NCC each node belongs to
        edge_classes = {}  # Store which NCC each edge belongs to
        
        # Assign nodes and edges to their NCCs
        for class_idx, component in enumerate(NCC):
            for node in component:
                node_classes.setdefault(node, []).append(class_idx)
            
            for edge in G.edges():
                if edge[0] in component and edge[1] in component:
                    edge_classes.setdefault(edge, []).append(class_idx)
        
        # Define colors:
        num_components = len(NCC)
        color_map = plt.cm.get_cmap("tab10", num_components)  # Use tab10 colormap
    
        node_colors = []
        node_labels = {}  # Store font colors for each node
        for node in G.nodes():
            if node not in node_classes:  
                node_colors.append("black")  # Uncategorized node
                node_labels[node] = "white"  # Set font color to white for black nodes
            elif len(node_classes[node]) > 1:  
                node_colors.append("gray")  # Shared node
                node_labels[node] = "black"  # Default font color
            else:  
                node_colors.append(color_map(node_classes[node][0]))  
                node_labels[node] = "black"  # Default font color
    
        edge_colors = []
        for edge in G.edges():
            if edge not in edge_classes:  
                edge_colors.append("black")  
            elif len(edge_classes[edge]) > 1:  
                edge_colors.append("gray")  
            else:  
                edge_colors.append(color_map(edge_classes[edge][0]))  
    
        # Create figure and adjust layout
        fig, ax = plt.subplots(figsize=(7, 5))
        pos = nx.spring_layout(G, seed=42)  
        nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, 
                node_size=600, edge_cmap=color_map, font_size=10, ax=ax)
        
        # Draw labels with appropriate font color
        for node, (x, y) in pos.items():
            plt.text(x, y, str(node), ha='center', va='center', 
                     color=node_labels[node])
    
        if (N == -1):
            N = "N"
        else:
            N = str(N)
        
        # Add title
        plt.title("Graph with " + N + "-connected Components\n(Gray = Shared, Black = Uncategorized)")
    
        # Adjust margins to make space for the text (left margin for separation)
        plt.subplots_adjust(left=0.3)  # Increase left margin to create space
    
        # Format the NCC list with line wrapping
        formatted_ncc_list = []
        for i, comp in enumerate(NCC):
            comp_str = N + f"CC {i+1}: {sorted(list(comp))}"
            wrapped_lines = textwrap.wrap(comp_str, width=max_line_length)
            formatted_ncc_list.extend(wrapped_lines)
    
        # Move the text to the left and ensure proper spacing
        subtitle_text = "\n".join(formatted_ncc_list)
        plt.figtext(0.02, 0.5, subtitle_text, wrap=True, horizontalalignment='left', 
                    verticalalignment='center', fontsize=10)
    
        plt.show()
    
