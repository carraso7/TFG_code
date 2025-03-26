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

class Triconnected_finder():
    
    def __find_sep_pairs(G):
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
    
    def __find_connected_components(G, sep_pairs):
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
    
    def __find_relation_R(G, connected_components):
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
    
    
    def __find_relation_T(G, relation_R):
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
    
    def __find_triply_connected_from_T_R(G, rel_T, rel_R):
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
    
    def find_triply_connected_comps(G):
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


class connected_components_drawer(): ### TODO LEER SI EN PYTHON ES CORRECTO PONER MÁS DE UNA CLASE EN EL MISMO ARCHIVO O DEBEN ESTAR EN ARCHIVOS SEPARADOS. 
### TODO ESCRIBIR A NETWORKX PARA VER SI QUIEREN MI LIBRERÍA