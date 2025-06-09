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
        for sp in list(combinations(G.nodes, 2)): # O(n^2) iterations
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
        Finds the relation R for each pair of nodes in 
        graph `G`. 

        Parameters
        ----------
        G : networkx.Graph
            Graph to find relation R.
        connected_components : dict
            Dictionary representing the connected components of the graph 
            without each separation pair (described in 
            `__find_connected_components`).

        Returns
        -------
        relation_R : list
            List of size `G.nodes()` * `G.nodes()`. Nodes with indexes i and j
            are R-related if and only if `relation_R[i][j]` is true.
            
        """
        node_index = {node: i for i, node in enumerate(G.nodes)}
        relation_R = [[True for _ in range(len(G.nodes))] for _ in range(len(G.nodes))]
        for pair in list(combinations(G.nodes, 2)):
            related = True
            for sep_pair in connected_components.keys():
                if (pair[0] not in sep_pair) and (pair[1] not in sep_pair):
                    if connected_components[sep_pair][pair[0]] != connected_components[sep_pair][pair[1]]:
                        related = False
            relation_R[node_index[pair[0]]][node_index[pair[1]]] = related
            relation_R[node_index[pair[1]]][node_index[pair[0]]] = related
                
        return relation_R
    
    
    def __find_relation_T(self, G, relation_R):
        """
        Find relation T for graph G.  

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
        node_index = {node: i for i, node in enumerate(G.nodes)}
        relation_T = []
        for trio in list(combinations(G.nodes, 3)):
            if relation_R[node_index[trio[0]]][node_index[trio[1]]] and relation_R[node_index[trio[0]]][node_index[trio[2]]] and relation_R[node_index[trio[2]]][node_index[trio[1]]]:
                relation_T.append(trio)
        return relation_T
    
    
    def __find_triply_connected_from_T_R(self, G, rel_T, rel_R, sep_pairs):
        """
        Function to find triply connected components from relations T and R of 
        graph ´G´

        Parameters
        ----------
        G : networkx.Graph
            Graph to find triconnected components.
        rel_T : list
            Relation T of ´G´ as described in ´__find_relation_T´.
        rel_R : list
            Relation R of ´G´ as described in ´__find_relation_R´.

        Returns
        -------
        list
            List of dictionaries with two entries. 'node list' and 
            'virtual edges'. Each dictionary represents a triconnected block
            with all its nodes in node list and its virtual edges in 
            'virtual edges'. Virtual edges are edges of the TCC that are not 
            contained in G
            
        """
        node_index = {node: i for i, node in enumerate(G.nodes)}
        triply_components = []
        for rel_T_elem in rel_T: 
            triply_component = list(rel_T_elem) 
            for node in G.nodes:
                if rel_R[node_index[rel_T_elem[0]]][node_index[node]] and rel_R[node_index[rel_T_elem[1]]][node_index[node]] and rel_R[node_index[rel_T_elem[2]]][node_index[node]]:
                    triply_component.append(node) 
            triply_components.append(triply_component)
        # Delete repeated triply connected components
        TCCs_lists = list(set(frozenset(comp) for comp in triply_components))
        TCCs = []
        for tcc_list in TCCs_lists:
            tcc = {"node_list": tcc_list, "virtual_edges": []}
            for sep_pair in sep_pairs:
                if (sep_pair[0] in tcc_list) and (sep_pair[1] in tcc_list) and (sep_pair not in G.edges()):  
                    tcc["virtual_edges"].append(sep_pair)    
            TCCs.append(tcc)
        return TCCs 
    
    
    def triconnected_comps(self, G):
        """
        Get triconnected components of graph `G` 
        
        Parameters
        ----------
        G : networkx.Graph
            Graph to find triconnected components.

        Returns
        -------
        TCCs : list of dictionaries            
            List of dictionaries with two entries. 'node list' and 
            'virtual edges'. Each dictionary represents a triconnected block
            with all its nodes in node list and its virtual edges in 
            'virtual edges'. Virtual edges are edges of the TCC not in G
        info: dictionary with the following string entries: 
            - 'all_relation_T' : list of lists with tuples of length 3
                One list for each biconnected component containing the 
                elements of relation T in that component represented by
                tuples of three nodes.
            - 'all_relation_R' : list of n*n matrices of booleans.
                One matrix for each biconnected component. Each matrix 
                represents the relation R between all the nodes, true if they 
                are related and false otherwise. The matrices are symmetric. 
            - 'all_connected_components' : list of dictionaries.
                One dictionary for each biconnected component. The keys of the 
                dictionary are the separation pairs and the value is a 
                dictionary with all the nodes labeled depending on their 
                connected components on the graph taking out the separation 
                pair. 
            - 'all_sep_pairs' : list of lists of tuples with length 2
                One list for each biconnected component. Each list contains  
                the separation pairs of the biconnected component.
                
        """
        all_TCCs = []
        all_relation_T = []
        all_relation_R = []
        all_connected_components = []
        all_sep_pairs = []
        
        # Find biconnected components and create subgraphs
        bicomponents = list(nx.biconnected_components(G))
        subgraphs = [G.subgraph(component).copy() for component in bicomponents if len(component) >= 3]
        
        for subgraph in subgraphs:
            sep_pairs = self.__find_sep_pairs(subgraph)
            connected_components = self.__find_connected_components(subgraph, sep_pairs)
            relation_R = self.__find_relation_R(subgraph, connected_components)
            relation_T = self.__find_relation_T(subgraph, relation_R)
            TCCs = self.__find_triply_connected_from_T_R(subgraph, relation_T, relation_R, sep_pairs)
            
            all_TCCs.extend(TCCs)
            all_relation_T.append(relation_T)
            all_relation_R.append(relation_R)
            all_connected_components.append(connected_components)
            all_sep_pairs.append(sep_pairs)
        
        info = {}
        info["relation_T"] = all_relation_T
        info["relation_R"] = all_relation_R
        info["connected_components"] = all_connected_components
        info["sep_pairs"] = all_sep_pairs
        
        return all_TCCs, info
    