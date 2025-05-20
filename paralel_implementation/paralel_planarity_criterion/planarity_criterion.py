"""
Created on Wed Mar 26 00:35:41 2025

@author: carlo
"""

import networkx as nx
import random
from itertools import combinations, permutations
import SAT2_solver
import triconnected_components as TCC

class PlanarityCriterion:  # TODO Pensar si dejar como clase

    def spanning_tree(self, G): 
        """
        Gets an spanning tree as an undirected graph from a random node

        Parameters
        ----------
        G : networkx.Graph 
            Graph to get spanning tree.

        Returns
        -------
        spanning_tree_undirected : networkx.Graph 
            spanning tree of `G` as undirected graph.

        """
        # Select a random starting node
        random_node = random.choice(list(G.nodes))

        # Get a spanning tree using DFS
        spanning_tree = nx.dfs_tree(G, source=random_node)

        # Convert spanning tree to an undirected graph
        spanning_tree_undirected = nx.Graph(spanning_tree) ### TODO Revisar si esto es necesario

        return spanning_tree_undirected

    
    def fundamental_cycles(self, G, spanning_tree_undirected): # TODO VER SI SE PUEDE HACER CON NETWORKX
        """
        Computes the fundamental cycles of a graph `G` with respect to a given 
        spanning tree. A fundamental cycle is formed by adding one non-tree 
        edge to the spanning tree.
    
        Parameters
        ----------
        G : networkx.Graph
            The original undirected graph.
        spanning_tree_undirected : networkx.Graph
            A spanning tree of `G`, represented as an undirected graph.
    
        Returns
        -------
        fundamental_cycles: list of lists
            A list of cycles, where each cycle is represented as a list of 
            nodes ending at the starting node to form a closed path.
            
        """
        # Ensure spanning tree edges are treated as undirected
        spanning_tree_edges = set(spanning_tree_undirected.edges())

        # Find edges not in the spanning tree (back edges)
        generating_edges = [(u, v) for u, v in G.edges() if (
            u, v) not in spanning_tree_edges and (v, u) not in spanning_tree_edges]

        # Find the fundamental set of cycles
        fundamental_cycles = []
        for u, v in generating_edges:
            # Find the path in the spanning tree between u and v
            path = nx.shortest_path(
                spanning_tree_undirected, source=u, target=v)
            cycle = path + [u]  # Complete the cycle
            fundamental_cycles.append(cycle)
        return fundamental_cycles
    
    
    def get_bridges(self, G, fundamental_cycles):  ### TODO TESTEAR ESTE MÉTODO DE BRIDGES CONTRA EL OTRO GET_BRIDGES_1
        
        """
        Identifies the bridges of `G` relative to each fundamental cycle 
        (defined in ### TODO REFERENCIA PAPER)
        
        Parameters
        ----------
        G : networkx.Graph
            Input graph.
        fundamental_cycles : list of lists
            A list of cycles forming a fundamental basis of cycles. Each cycle 
            is represented by a list of nodes ending at the starting node.
        
        Returns
        -------
        bridges_all_cycles: dict[tuple[Hashable], list[dict]]
            A dictionary where each key is a cycle (as a tuple of nodes with
            the same start and end node), and each value is a list of bridges. 
            Each bridge is represented as a dictionary with:
                - 'edges': list of tuples.
                    Edges (undirected )in the bridge as tuples of nodes.
                - 'att_ver': set.
                    Attachment vertices with the cycle as a list of nodes.
                
        """
        bridges_all_cycles = {}
        attachment_vertices_all_cycles = {}

        for c in fundamental_cycles: 
            attachment_vertices = [] ### TODO CREO QUE SE PUEDE HACER SIN ATT VERT GLOBAL
            attachment_edges = [] 
            bridges = []
            
            # Create a copy of the graph and remove cycle edges
            G_no_c_edges = G.copy()
            G_no_c_nodes = G.copy()
            G_no_c_edges.remove_edges_from(
                [(c[i], c[i+1]) for i in range(len(c)-1)])
            G_no_c_nodes.remove_nodes_from(c)
        
            for cycle_node in c[:-1]:  ### TODO REVISAR REPETICIONES AQUÍ.
                for att_edge in G_no_c_edges.edges(cycle_node):
                    # Attachment edge with one node outside of cycle
                    if att_edge[0] not in c or att_edge[1] not in c:
                        attachment_vertices.append(att_edge[0] if att_edge[0] not in c else att_edge[1])
                        # Add edge to attachment edges only if it has one
                        # node outside of the cycle
                        attachment_edges.append(att_edge) 
                    # Attachment edge with both nodes in the cycle (it forms a
                    # bridge on its own).
                    else: ### TODO INVERT THIS IF AND MAKE ELIF WITH NEXT CLAUSE
                        if (att_edge not in attachment_edges) and ((att_edge[1], att_edge[0]) not in attachment_edges):
                            bridge = {
                                "edges": [att_edge],
                                "att_ver": set([att_edge[0], att_edge[1]])
                            }
                            bridges.append(bridge)
                            attachment_vertices.extend([att_edge[0], att_edge[1]])
                            attachment_edges.append(att_edge)
        
            # Eliminate duplicates ### TODO HACE FALTA??
            attachment_vertices = list(set(attachment_vertices))
            attachment_vertices_all_cycles[tuple(c)] = attachment_vertices
        
            connected_comps = list(nx.connected_components(G_no_c_nodes))
            # Add attatchment edges to each component
            for comp in connected_comps:
                subgraph = G_no_c_nodes.subgraph(comp)
                bridge = {
                    "edges": list(subgraph.edges()),
                    "att_ver": set([])
                }
                for edge in attachment_edges:
                    u, v = edge
                    if (u in comp) or (v in comp):
                        bridge["edges"].append(edge)
                        bridge["att_ver"].add(u if u in c else v)
                if bridge["edges"]:
                    bridges.append(bridge)
                    
            bridges_all_cycles[tuple(c)] = bridges

        return bridges_all_cycles


    ### Auxiliar functions for getting 2 CNF conditions ###
    
    def __update_cond_a(
            self, G, fundamental_cycles, bridges_all_cycles, edge_index_map, 
            implications
            ):
        n_edges = len(G.edges())
        n_cycles = len(fundamental_cycles)
        general_neg_offset = n_edges * n_cycles
        for c_index, c in enumerate(fundamental_cycles):
            offset = c_index * n_edges
            neg_offset = general_neg_offset + c_index * n_edges
            for bridge in bridges_all_cycles[tuple(c)]:
                # Usamos directamente las aristas del puente
                for edge1, edge2 in combinations(bridge["edges"], 2): 
                    e1, e2 = edge_index_map[edge1], edge_index_map[edge2]
                    # e1 -> e2
                    implications.append((offset + e1, offset + e2))
                    # e2 -> e1
                    implications.append((offset + e2, offset + e1))
                    # ¬e1 -> ¬e2
                    implications.append((neg_offset + e1, neg_offset + e2))
                    # ¬e2 -> ¬e1
                    implications.append((neg_offset + e2, neg_offset + e1))
                    
        return implications


    def __get_edges_cycle(self, c):# TODO REVISAR SI AQUÍ SE AÑADEN LOS EDGES EN ORDEN CONTRARIO
        edges = []
        for i in range(len(c) - 1):
            edge = [c[i], c[i + 1]]
            edge.sort()
            edges.append(tuple(edge))
        return edges


    def __update_cond_b(self, G, fundamental_cycles, edge_index_map, 
                        cycle_index_map, implications
                        ):
        n_edges = len(G.edges())
        n_cycles = len(fundamental_cycles)
        neg_offset = n_edges * n_cycles
        # TODO CHEQUEAR BIEN ESTA CONDICIÓN
        for cycle1, cycle2 in combinations(fundamental_cycles, 2):
            c1_edges = self.__get_edges_cycle(cycle1)
            c2_edges = self.__get_edges_cycle(cycle2)
            c1 = cycle_index_map[tuple(cycle1)]
            c2 = cycle_index_map[tuple(cycle2)]
            c1_not_c2 = [edge for edge in c1_edges if edge not in c2_edges]
            c2_not_c1 = [edge for edge in c2_edges if edge not in c1_edges]
            for edge1 in c1_not_c2:
                for edge2 in c2_not_c1:
                    e1, e2 = edge_index_map[edge1], edge_index_map[edge2]
                    # e1c2 -> ¬e2c1
                    implications.append((c2*n_edges + e1, neg_offset + c1*n_edges + e2))
                    # e2c1 -> ¬e1c2
                    implications.append((c1*n_edges + e2, neg_offset + c2*n_edges + e1))
        return implications


    # Auxiliar functions for checking conflicts in condition c)

    def __conflict_type_1(self, bridge_pair, c):
        common_att_vert = 0
        for vertex in bridge_pair[0]["att_ver"]:
            if vertex in bridge_pair[1]["att_ver"]:
                common_att_vert += 1
                if common_att_vert >= 3:
                    return True  
        return common_att_vert >= 3

    # TODO checkear que los ciclos siempre entran aquí ordenados
    def __conflict_type_2(self, bridge_pair, c):
        matching_seq = 0

        # Look for the pattern starting on attachment vertices of both pairs
        for node in c[0:len(c) - 1]:
            if (node in bridge_pair[0]["att_ver"]) and matching_seq % 2 == 0:
                matching_seq += 1
                # print(matching_seq)
                if matching_seq >= 4:
                    return True
            elif (node in bridge_pair[1]["att_ver"]) and matching_seq % 2 == 1:
                matching_seq += 1
                # print(matching_seq)
                if matching_seq >= 4:
                    return True

        # Treat last node of the cycle differently. Otherwise, there can be errors if the starting node
        # of the cycle is of attachment of both bridges.
        if (c[len(c) - 1] in bridge_pair[1]["att_ver"]) and matching_seq % 2 == 1 and (c[len(c) - 1] not in bridge_pair[0]["att_ver"]):
            matching_seq += 1
            # print(matching_seq)
            if matching_seq >= 4:
                return True

        matching_seq = 0
        for node in c[0:len(c) - 1]:
            # print()
            # print(node)
            # print(matching_seq)
            if (node in bridge_pair[1]["att_ver"]) and matching_seq % 2 == 0:
                matching_seq += 1
                # print(matching_seq)
                if matching_seq >= 4:
                    return True
            elif (node in bridge_pair[0]["att_ver"]) and matching_seq % 2 == 1:
                matching_seq += 1
                # print(matching_seq)
                # print(matching_seq >= 4)
                if matching_seq >= 4:
                    return True  # TODO ALGUNOS DE ESTOS SOBRAN SEGÚN LOS MÓDULOS

        # Treat last node of the cycle differently. Otherwise, there can be 
        # errors if the starting node of the cycle is vertex of attachment 
        # of both bridges.
        if (c[len(c) - 1] in bridge_pair[0]["att_ver"]) and matching_seq % 2 == 1 and (c[len(c) - 1] not in bridge_pair[1]["att_ver"]):
            matching_seq += 1
            if matching_seq >= 4:
                return True

        return matching_seq >= 4

    def __conflict_between(self, bridge_pair, c):
        return self.__conflict_type_1(bridge_pair, c) or self.__conflict_type_2(bridge_pair, c)

    def __update_cond_c(
            self, G, fundamental_cycles, bridges_all_cycles, edge_index_map,
            implications
            ):
        n_edges = len(G.edges())
        n_cycles = len(fundamental_cycles)
        general_neg_offset = n_edges * n_cycles 
        for c_index, c in enumerate(fundamental_cycles):
            offset = c_index * n_edges
            neg_offset = general_neg_offset + c_index * n_edges
            for bridge1, bridge2 in combinations(bridges_all_cycles[tuple(c)], 2):
                if self.__conflict_between((bridge1, bridge2), c):
                    for edge1 in bridge1["edges"]:
                        for edge2 in bridge2["edges"]:
                            e1, e2 = edge_index_map[edge1], edge_index_map[edge2] 
                            # e1 -> ¬e2
                            implications.append((offset + e1, neg_offset + e2))
                            # e2 -> ¬e1
                            implications.append((offset + e2, neg_offset + e1))
                            # ¬e1 -> e2
                            implications.append((neg_offset + e1, offset + e2))
                            # ¬e2 -> e1
                            implications.append((neg_offset + e2, offset + e1))
        return implications

    
    def __get_implications_2CNF(
            self, G, fundamental_cycles, bridges_all_cycles, edge_index_map,
            cycle_index_map
            ):
        """
        Function that returns the list of implications derived from the clauses
        of the 2CNF problem of getting a sattisfying truth assigment of graph
        G using all the input information of the graph. 
        
        The implications are represented as tuples of index in range(2 * c * m),
        where c is the number of fundamental cycles and m is the number of 
        edges in `G`. Each integer of the tuple represents a variable of the 
        2CNF problem or a negated variable of the problem. The integer that 
        represents the variable with cycle index x and edge index y is 
        x * m + y and the corresponding negated variable is x * m + y + c * m. 
        Note that the numbers in the range whose edge belongs to the cycle do 
        not represent any variable.

        Parameters ### TODO EXPLICAR PARAMETROS O QUITAR SECCIÓN
        ----------
        G : TYPE
            DESCRIPTION.
        fundamental_cycles : TYPE
            DESCRIPTION.
        bridges_all_cycles : TYPE
            DESCRIPTION.
        edge_index_map : TYPE
            DESCRIPTION.
        cycle_index_map : TYPE
            DESCRIPTION.

        Returns
        -------
        implications : list of tuples with integers.
            Implications are represented as tuples of index in range(2*c*m),
            where c is the number of fundamental cycles and m is the number of 
            edges in `G`. Each integer of the tuple represents a variable of 
            the 2CNF problem or a negated variable of the problem. The integer 
            that represents the variable with cycle index x and edge index y is 
            x*m + y and the corresponding negated variable is x*m + y + c*m. 
            Note that the numbers in the range whose edge belongs to the cycle 
            do not represent any variable.
            
        """
        implications = []
        
        implications = self.__update_cond_a(
            G, fundamental_cycles, bridges_all_cycles, edge_index_map, 
            implications
            )
        
        implications = self.__update_cond_b(
            G, fundamental_cycles, edge_index_map, cycle_index_map, 
            implications
            )
        
        implications = self.__update_cond_c(
            G, fundamental_cycles, bridges_all_cycles, edge_index_map, 
            implications
            )
        
        return implications
    
    
    def get_truth_assigment(self, G, fundamental_cycles, bridges_all_cycles):
        """
        Get a satisfiying truth assigment of variables indexed by fundamental
        cycles and edges of G. Each variable is indexed by a number in 
        range(c * m), where c is the number of fundamental cycles and m is the 
        number of edges in `G`. The integer represents the variable with cycle 
        index x and edge index y. Note that the numbers in the range whose edge 
        belongs to the cycle do not represent any variable.

        Parameters ### TODO EXPLICAR PARAMETROS O QUITAR SECCIÓN
        ----------
        G : TYPE
            DESCRIPTION.
        fundamental_cycles : TYPE
            DESCRIPTION.
        bridges_all_cycles : TYPE
            DESCRIPTION.

        Returns
        -------
        results : list of booleans. 
            List of length number of fund. cycles * number of bridges.
            
            Each variable is indexed by a number in range(c * m), where c is
            the number of fundamental cycles and m is the number of edges in 
            `G`. The integer represents the variable with cycle index x and 
            edge index y. Note that the numbers in the range whose edge belongs
            to the cycle do not represent any variable.
        info : ### TODO EXPLICAR PARAMETROS O QUITAR SECCIÓN

        """
        
        edge_index_map = {}
        for i, edge in enumerate(G.edges()):
            edge_index_map[edge] = i
            # Store reversed edge as well ### TODO INTENTAR NO USAR ESTAS LÍNEAS
            edge_index_map[(edge[1], edge[0])] = i
            
            cycle_index_map = {}
            index_cycle_map = {}
            for i, c in enumerate(fundamental_cycles):
                cycle_index_map[tuple(c)] = i  # tuple directly
                index_cycle_map[i] = c  
                
        # n_cycle*n_edges + n_edge: positive variable
        # n_cycle*n_edges + n_edge + n_cycles*n_edges:negative variable
        implications = self.__get_implications_2CNF(
            G, fundamental_cycles, bridges_all_cycles, edge_index_map, 
            cycle_index_map
            )
        solver = SAT2_solver.SAT2_solver()
        n_variables = len(fundamental_cycles) * len(G.edges())
        results, info = solver.get_truth_assigment(implications, n_variables)
        info["edge_index_map"] = edge_index_map
        info["cycle_index_map"] = cycle_index_map
        info["index_cycle_map"] = index_cycle_map
        return results, info 
            
    
    def compute_lt(self, G, truth_assign, fundamental_cycles, info):
        """
        Computes the strict containment relation (<) between fundamental cycles based
        on a truth assignment from a 2-SAT formula. One cycle c1 is strictly
        contained in another cycle c2 if it is contained and there is no other
        cycle c3 such that c1 is contained in c3 and c3 is contained in c2. 
        A cycle is contained in another cycle if it has an edge contained in 
        the other cycle according to the truth assigment. 
        
    
        Parameters
        ----------
        G : networkx.Graph
            The input graph.
        truth_assign : list of booleans
            Boolean values indicating the inclusion status of each edge in each 
            cycle.
        fundamental_cycles : list of lists of integers
            List of fundamental cycles in node form.
    
        Returns
        -------
        rel_lt : n * n matrix, where n is the number of fund. cycles.
            rel_lt[i][j] = 1 if cycle j < cycle i,
        info: dict ### TODO EXPLICAR PARAMETROS O QUITAR SECCIÓN
        
        """
        rel_lt = [[0 for _ in range(len(fundamental_cycles))] for _ in range(len(fundamental_cycles))]
        rel_in = [[0 for _ in range(len(fundamental_cycles))] for _ in range(len(fundamental_cycles))]
        cycle_index_map = info["cycle_index_map"]
        edge_index_map = info["edge_index_map"]
        
        # Compute first contained relation (not strictly)
        # We do combinations because we check both ways
        for cycle1, cycle2 in combinations(fundamental_cycles, 2): 
            c1_edges = self.__get_edges_cycle(cycle1)
            c2_edges = self.__get_edges_cycle(cycle2)
            c1 = cycle_index_map[tuple(cycle1)]
            c2 = cycle_index_map[tuple(cycle2)]
            for edge in c1_edges:
                if edge not in c2_edges:    
                    c1_not_c2 = edge_index_map[edge]
                    break
            for edge in c2_edges:
                if edge not in c1_edges:
                    c2_not_c1 = edge_index_map[edge]
                    break
            # check both ways
            if truth_assign[c2*len(G.edges()) + c1_not_c2]:
                rel_in[c1][c2] = 1
            if truth_assign[c1*len(G.edges()) + c2_not_c1]:
                rel_in[c2][c1] = 1
        
        for cycle1, cycle2 in permutations(fundamental_cycles, 2): ### TODO SEGURAMENTE ES MEJOR CON COMBINATIONS Y PROBANDO EN DISTINTO ORDEN
            c1_edges = self.__get_edges_cycle(cycle1)
            c2_edges = self.__get_edges_cycle(cycle2)
            c1 = cycle_index_map[tuple(cycle1)]
            c2 = cycle_index_map[tuple(cycle2)]
            if rel_in[c1][c2]: # if its value is 1
                rel_lt[c1][c2] = 1
                # Check if there is any cycle between c1 and c2
                for cycle3 in fundamental_cycles:    
                    c3 = cycle_index_map[tuple(cycle3)]
                    if rel_in[c1][c3] and rel_in[c3][c2]:
                        rel_lt[c1][c2] = 0
                        break
                    if rel_lt[c1][c2] == 0:
                        break
                    
        info["rel_in"] = rel_in
        return rel_lt, info
               
        
    def __sum_cycles(self, edges1, edges2): ##################TODO TODO TODO TODO REVISAR ESTE ################################
        """
        Compute the symmetric difference between two cycles represented as lists of edges.
    
        Parameters
        ----------
        edges1 : list of tuple
            List of edges (each edge is a tuple of two vertices) representing the first cycle.
        edges2 : list of tuple
            List of edges (each edge is a tuple of two vertices) representing the second cycle.
    
        Returns
        -------
        list of tuples
            A list of edges representing the symmetric difference of the two cycles.
            These are the edges that appear in exactly one of the two input cycles.
        """

        set1 = set(edges1)
        set2 = set(edges2)
        return list(set1 ^ set2)  # edges in one or the other, but not both
    
    def edges_to_cycle(self, edges):  ##################TODO TODO TODO TODO REVISAR ESTE ################################
        """
        Reconstruct a single simple cycle from a set of edges.
    
        Parameters
        ----------
        edges : iterable of tuple
            A set or list of edges (each edge is a tuple of two vertices).
    
        Returns
        -------
        list
            A list of vertices representing the reconstructed cycle.
            The first vertex is repeated at the end to close the cycle.
    
        Raises
        ------
        ValueError
            If the input edges do not form exactly one simple cycle.
            This includes cases where:
            - Some vertices do not have degree exactly 2.
            - The edges form multiple disjoint cycles or an open path.
            - The graph is not connected or has extra edges.
        """
        if not edges:
            raise ValueError("No edges provided.")
    
        # Build adjacency map
        adj = {}
        for u, v in edges:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
    
        # Check: every vertex in a simple cycle has degree exactly 2
        for vertex, neighbors in adj.items():
            if len(neighbors) != 2:
                raise ValueError(f"Vertex {vertex} does not have degree 2; not a simple cycle.")
    
        # Reconstruct the cycle
        start = next(iter(adj))
        cycle = [start]
        current = start
        prev = None
    
        while True:
            neighbors = adj[current]
            # Choose the next neighbor that's not the previous vertex
            next_vertex = neighbors[0] if neighbors[0] != prev else neighbors[1]
            
            if next_vertex == start:
                break  # cycle closed
    
            cycle.append(next_vertex)
            prev, current = current, next_vertex
    
        cycle.append(start)  # close the cycle
    
        # Validate: have we visited all edges?
        if len(cycle) - 1 != len(edges):
            raise ValueError("Edges do not form a single cycle (multiple components or extra edges).")
    
        return cycle

    
    def get_peripheral_basis(self, rel_lt, fundamental_cycles, info):
        """
        Constructs a peripheral cycle basis from the fundamental cycles using 
        the strict containment relation between them.
    
        Each peripheral cycle is computed by summing a fundamental cycle with 
        all cycles strictly contained in it, according to the `rel_lt` matrix.
    
        Parameters
        ----------
        rel_lt : list[list[int]]
            A square binary matrix where rel_lt[i][j] == 1 means that cycle j
            is strictly contained in cycle i.
        fundamental_cycles : list[list[Hashable]]
            The list of fundamental cycles, each as a sequence of nodes.
        info : dict
            Dictionary containing:
                - 'cycle_index_map': dict[tuple[Hashable], int]
                - 'index_cycle_map': dict[int, list[Hashable]]
    
        Returns
        -------
        periph_basis_edges: list of tuples of ints.
            List of lists of edges. Each list has the edges of a peripheral
            cycle as edges. 
        info: dict
        
        """
        periph_basis_lists = []
        periph_basis_edges = []
        cycle_index_map = info["cycle_index_map"]
        index_cycle_map = info["index_cycle_map"]
        for cycle1 in fundamental_cycles:
            periph_cycle = self.__get_edges_cycle(cycle1)
            c1 = cycle_index_map[tuple(cycle1)]
            for c2 in range(len(fundamental_cycles)):
                if rel_lt[c2][c1]:
                    cycle2 = self.__get_edges_cycle(index_cycle_map[c2])
                    periph_cycle = self.__sum_cycles(periph_cycle, cycle2)
            periph_basis_lists.append(self.edges_to_cycle(periph_cycle))
            periph_basis_edges.append(periph_cycle)
        info["periph_basis_lists"] = periph_basis_lists 
        return periph_basis_edges, info
    
    
    def get_plane_mesh(self, periph_basis_edges):
        """
        Builds the full plane mesh by adding the outer face to the peripheral 
        cycle basis.
    
        The outer face is computed as the symmetric difference of all 
        peripheral cycles.
    
        Parameters
        ----------
        periph_basis_edges: list of tuples of ints.
            List of lists of edges. Each list has the edges of a peripheral
            cycle as edges. 
    
        Returns
        -------
        plane_mesh: list of tuples of ints
            The plane mesh, consisting of all peripheral cycles plus the outer 
            face, each represented as a list of undirected edges.
            
        """
        outer_cycle =  []
        for cycle in periph_basis_edges:
            outer_cycle = self.__sum_cycles(outer_cycle, cycle)
        plane_mesh = periph_basis_edges.copy()
        plane_mesh.append(outer_cycle)
        return plane_mesh
    
    
    def is_planar(self, G): ### TODO GESTIONAR BIEN LO DE LA INFO PARA CADA COMPONENTE
        """
        Determines if the graph G is planar using Mac Lane criterion as stated
        in ### TODO INCLUIR REFERENCIA

        Parameters
        ----------
        G : networkx.Graph
            Graph to determine its planarity.

        Returns
        -------
        bool
            Boolean determining if the graph is planar.
        info: dict
            Dictionary with extra information about the algorithm

        """
        # If the graph has more than 3*n - 6 edges, return False. ### TODO REFERENCIA DEMO DE ESTO
        if (len(G.edges()) > 3 * len(G.nodes()) - 6):
            return False, None 
        
        # Analyze each triconnected component individually. The graph is planar
        # if all its components are. 
        finder = TCC.TriconnectedFinder()
        TCCs, info = finder.triconnected_comps(G)
        info["TCCs"] = TCCs
        info["planarity_info"] = []
        for tcc_list in TCCs:
            TCC_info = {}
            tcc = G.subgraph(tcc_list["node_list"]).copy()

            # Add virtual edges to each tcc
            tcc.add_edges_from(tcc_list["virtual_edges"])
            
            spanning_tree = self.spanning_tree(tcc)
            
            fundamental_cycles = self.fundamental_cycles(tcc, spanning_tree)
            
            bridges = self.get_bridges(tcc, fundamental_cycles)
            
            truth_assign, TCC_info = self.get_truth_assigment(
                tcc, fundamental_cycles, bridges
                )
            
            # Init info dictionary
            TCC_info["spanning_tree"] = spanning_tree
            TCC_info["fundamental_cycles"] = fundamental_cycles
            TCC_info["bridges"] = bridges
            TCC_info["truth_assign"] = truth_assign
            TCC_info["rel_in"] = "No info"
            info["Failing_tcc"] = "No info"
            info["failing_reason"] = "No info"
            info["failing edge"] = "No info"
            
            if truth_assign is None:
                info["failing_tcc"] = tcc
                info["failing_reason"] = "No truth assignment"
                info["planarity_info"].append(TCC_info)
                return False, info
            
            rel_lt, TCC_info = self.compute_lt(
                tcc, truth_assign, fundamental_cycles, TCC_info
                )
            TCC_info["rel_lt"] = rel_lt
            
            peripheral_basis, TCC_info = self.get_peripheral_basis(
                rel_lt, fundamental_cycles, TCC_info
                )     
            TCC_info["periph_basis"] = peripheral_basis
        
            plane_mesh = self.get_plane_mesh(peripheral_basis)
            TCC_info["plane_mesh"] = plane_mesh
            
            edges_count = {edge: 0 for edge in tcc.edges()}
            TCC_info["edges_count"] = edges_count
            for cycle in plane_mesh:
                for edge in cycle:
                    if edge in edges_count:
                        edges_count[edge] += 1
                    elif (edge[1], edge[0]) in edges_count:
                        edges_count[(edge[1], edge[0])] += 1
                    else: 
                        #print(cycle)  ### TODO PRINT QUITAR ### TODO REVISAR ESTO BIEN CON EL PAPER  Y QUE ES NECESARIO
                        info["planarity_info"].append(TCC_info)
                        return False, info
                    
            for edge, count in edges_count.items():
                if count != 2:
                    info["failing tcc"] = tcc
                    info["failing edge"] = edge
                    info["failing_reason"] = "Bad plane mesh"
                    TCC_info["edges_count"] = edges_count
                    info["planarity_info"].append(TCC_info)
                    return False, info
                
            info["planarity_info"].append(TCC_info)
            
        return True, info
    
   
"""   
   
   
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
### TODO HAY AQUÍ UN PRINTER QUE FALTA POR PONER de cnf_lists_by_cycle


#### CLASS CNF SOLVER ANTIGUA PARA USAR CON LAS LISTAS ########################


class CNF2Solver:
    
    def __get_adding_edges(self, prev_CNF_lists, CNF_lists, edge, cycle,
                           n_list, edge_index_map, cycle_index_map
                           ): ## TODO Hacerlo con los tres tipos de lista
        adding_edges = set([])
        # P => P
        if n_list == 0:
            for edge_cycle in CNF_lists[edge_index_map[edge]][cycle_index_map[tuple(cycle)]][n_list]:
                adding_edges.update(prev_CNF_lists[edge_index_map[edge_cycle[0]]][edge_cycle[1]][n_list]) ## TODO apañar aquí que no se haga con el id del cycle
            for edge_cycle in CNF_lists[edge_index_map[edge]][cycle_index_map[tuple(cycle)]][1]:
                adding_edges.update(prev_CNF_lists[edge_index_map[edge_cycle[0]]][edge_cycle[1]][2])
        
        # P => N
        if n_list == 1: 
            # P => N => N
            for edge_cycle in CNF_lists[edge_index_map[edge]][cycle_index_map[tuple(cycle)]][n_list]:
                adding_edges.update(prev_CNF_lists[edge_index_map[edge_cycle[0]]][edge_cycle[1]][3]) ## TODO apañar aquí que no se haga con el id del cycle
            # P => P => N
            for edge_cycle in CNF_lists[edge_index_map[edge]][cycle_index_map[tuple(cycle)]][0]:
                adding_edges.update(prev_CNF_lists[edge_index_map[edge_cycle[0]]][edge_cycle[1]][1])
                
        # N => P
        if n_list == 2: 
            # N => P => P
            for edge_cycle in CNF_lists[edge_index_map[edge]][cycle_index_map[tuple(cycle)]][n_list]:
                adding_edges.update(prev_CNF_lists[edge_index_map[edge_cycle[0]]][edge_cycle[1]][0]) ## TODO apañar aquí que no se haga con el id del cycle
            # N => N => P
            for edge_cycle in CNF_lists[edge_index_map[edge]][cycle_index_map[tuple(cycle)]][3]:
                adding_edges.update(prev_CNF_lists[edge_index_map[edge_cycle[0]]][edge_cycle[1]][2])
                
        # N => N
        if n_list == 3: 
            # N => N => N
            for edge_cycle in CNF_lists[edge_index_map[edge]][cycle_index_map[tuple(cycle)]][n_list]:
                adding_edges.update(prev_CNF_lists[edge_index_map[edge_cycle[0]]][edge_cycle[1]][n_list]) ## TODO apañar aquí que no se haga con el id del cycle
            # N => P => N
            for edge_cycle in CNF_lists[edge_index_map[edge]][cycle_index_map[tuple(cycle)]][2]:
                adding_edges.update(prev_CNF_lists[edge_index_map[edge_cycle[0]]][edge_cycle[1]][1])
        
        return adding_edges ### TODO REVISAR RESULTADOS Y HACER PARA CUALQUIER N_LIST. Probar si con más iteraciones no cambia la lista
    
    def update_CNF_iterative(self, CNF_lists, G, fundamental_cycles, 
                             edge_index_map, cycle_index_map
                             ):
        if len(fundamental_cycles) == 0: ### TODO SEGURAMENTE SE PUEDA QUITAR SI ASEGURAMOS TRICONNECTED
            return CNF_lists
        for i in range(math.ceil(math.log2(len(fundamental_cycles) * len(G.edges())))): ### TODO QUITAR ESTO Y EXPLICAR Y DEMOSTRAR PQ HAY QUE PONER EL LOG
            prev_CNF_lists = CNF_lists.copy()
            for cycle in fundamental_cycles:
                for edge in G.edges():
                    for n_list in range(4):
                        adding_edges = self.__get_adding_edges(
                            prev_CNF_lists, CNF_lists, edge, cycle, n_list, 
                            edge_index_map, cycle_index_map
                            )
                        # if len(adding_edges ) > 1: ###
                        #     print("adding",adding_edges)
                        #     print(CNF_lists[edge_index_map[edge]][cycle_index_map[tuple(cycle)]][i])
                        CNF_lists[edge_index_map[edge]][cycle_index_map[tuple(cycle)]][n_list].update(adding_edges)
                        # if len(adding_edges ) > 1:print(CNF_lists[edge_index_map[edge]][cycle_index_map[tuple(cycle)]][i]) ###
        
                        # for adding_edge in adding edges:
                        #     CNF_lists[edge_index_map(edge)][cycle_index_map(tuple(cycle))][i].add(adding_edge) ## TODO VER SI SE PUEDE AÑADIR UNA LISTA DE ELEMENTOS

        return CNF_lists
    
    def solve_CNF_from_1_list(self, CNF_lists):
        ### TODO DEMOSTRAR SI SE PUEDE HACER CON UNA ÚNICA LISTA Y SI SE PUEDE, 
        ### REFERENCIAR EN LOS COMENTARIOS.
        pass
     
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
#####################################################################################################   
 



"""


   
"""

### TODO IR TRASPASANDO SIGUIENTES PASOS


### Get pseudoembedding from CNF list ###

pseudoembedding = [[None for _ in range(len(fundamental_cycles))] for _ in range(len(G.edges()))]

def expand_bools(new_pseudo, CNF_lists, edge, cycle):
    for b in [True, False]:
        new_pseudo[edge_index_map[edge]][cycle_index_map[tuple(cycle)]] = b
        for variable in CNF_lists[][][if b 0, 2 ; 1, 3]
            if 
            
        
        
def set_bool(pseudoembedding, CNF_lists, edge, cycle):
    new_pseudo = pseudoembedding.copy()
    correct_pseudo = True
    info = {}
    # new_pseudo[edge_index_map[edge]][cycle_index_map[tuple(cycle)]] = b
    new_pseudo, info = expand_bools(new_pseudo, CNF_lists, edge, cycle)
    return new_pseudo, info # New_pseudo can be None if there is no pseudoembedding
        

for cycle_i, cycle in enumerate(fundamental_cycles):
    for edge_i, edge in enumerate(G.edges()):
        if edge not in cycle and pseudoembedding[edge_i][cycle_i] is None:
            set_bool(edge, cycle)
            
            
            
### Compute relation < and contained ###

contained_lists = [[] for _ in range len(fundamental_cycles)] # List of cycles contained on each cycle
def contained(cycle1, cycle2 pseudoembedding): # True if cycle1 contained in cycle2
    if cycle1 == cycle2:
        return False
    difference = list(set(get_edges_cycle(cycle1)) - set(get_edges_cycle(cycle2)))
    return pseudoembedding[edge_index_map(difference[0])] [cycle_index_map(cycle2)]

for cycle1 in fundamental_cycles:
    for cycle2 in fundamental_cycles:
        if contained(cycle_1, cycle2):
            contained_lists[cycle_index_map(cycle2)].append(cycle1) # cycle 1 contained in cycle 2
            
strictly_contained_lists = []
for contained_list in contained_lists:
    strictly_contained = []
    for cycle1 in contained_list:
        s_c = True
        for cycle2 in contained_list:
            if contained(cycle1, cycle2):
                s_c = False
                break
        if s_c:
            strictly_contained.append(cycle1)
    strictly_contained_lists.append(strictly_contained) ### TODO comprobar que siempre se añaden en ordden
            
        
    
### computar la relación contenido len(fundamental_cycles) * len(fundamental_cycles) y luego ir hacia detrás en cada ciclo para ver los ciclos <
    
       
    







### Obtain new peripheral cycle basis ###

def get_nodes_cyle(edges): ### TODO dada por chatGPT. Revisar
    from collections import defaultdict

    # Construct the adjacency list
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    visited = set()
    parent = {}
    
    def dfs(node, par):
        visited.add(node)
        parent[node] = par
        for neighbor in graph[node]:
            if neighbor == par:  # Ignore the edge to parent
                continue
            if neighbor in visited:  # Cycle detected
                cycle = []
                cur = node
                while cur != neighbor:
                    cycle.append(cur)
                    cur = parent[cur]
                cycle.append(neighbor)
                cycle.append(node)  # Close the cycle
                return cycle[::-1]  # Reverse for correct order
            else:
                result = dfs(neighbor, node)
                if result:
                    return result
        return None

    # Try to find a cycle from any node
    for node in graph:
        if node not in visited:
            cycle = dfs(node, None)
            if cycle:
                return cycle

    return None  # No cycle found

def update_cycle(c1, strictly_contained_list): ### TODO _PENSAR SI LLAMAR A ESTO SUMA DE CICLOS
    c1_edges = get_edges_cycle(c1)
    new_cycle_edges = c1_edges.copy()
    for c2 in strictly_contained_list:
        c2_edges = get_edges_cycle(c2)
        for edge in c2_edges:
            if edge in c1_edges:
                new_cycle_edges.delete(edge) ### TODO Creo que está mal por que no suma todos con todos sino todos con el primero
            else:
                new_cycle_edges.append(edge)
    return get_nodes_cycle(new_cycle_edges)

peripheral_cycle_basis = []
for c1_index, c1 in enumerate(fundamental_cycle_basis):
    new_cycle = update_cycle(c1, strictly_contained_lists[c1_index])
    peripheral_cycle_basis.append(new_cycle)
    
    
    


###Obtain plane mesh and check planarity ###

last_cycle = []
for c in peripheral_cycle_basis:
    sum_cycles(last_cycle, c)
plane_mesh = peripheral_cycle_basis.copy()
plane_mesh.append(last_cycle)

def is_planar(plane_mesh) ### TODO este algorigmo vale solo para plane meshes que vengan de grafos 3 conectados (o quizá dos).
    edge_counts = [0 for _ in range(len(G.edges()))]
    for c in plane_mesh:
        for edge in get_edges_cycle(c):
            edge_counts[edge_index_map(edge)] += 1
    for count in edge_counts:
        if count != 2:
            return False
    return True

print("Graph G planar: ", is_planar(plane_mesh))

"""


    
"""        
### FUNCIONES ANTIGUAS DE 2CNF PARA HACERLO CON LISTAS COMO EN EL PAPER CON LA CLASE DEL FINAL DEL ARCHIVO ###        
#####################################################################################################################        
        
    ### Auxiliar functions for getting 2 CNF conditions ###

    def __update_cond_a1(
            self, G, fundamental_cycles, bridges_all_cycles, edge_index_map, 
            CNF_lists
            ):
        for c_index, c in enumerate(fundamental_cycles):
            for bridge in bridges_all_cycles[tuple(c)]:
                # Usamos directamente las aristas del puente
                for edge1, edge2 in combinations(bridge["edges"], 2):
                    # TODO PENSAR SI ES MEJOR METER ÍNDICES  AÑADIR CON RESPECTO A CICLO
                    e1, e2 = edge_index_map[edge1], edge_index_map[edge2]
                    # CNF_lists[e1][c_index][0].add(e2)
                    # CNF_lists[e2][c_index][0].add(e1)
                    # CNF_lists[e1][c_index][3].add(e2)
                    # CNF_lists[e2][c_index][3].add(e1)
                    CNF_lists[e1][c_index][0].add((edge2, c_index))
                    # TODO PENSAR SI ES MEJOR METER ÍNDICES dentro de las listas
                    CNF_lists[e2][c_index][0].add((edge1, c_index))
                    CNF_lists[e1][c_index][3].add((edge2, c_index))
                    CNF_lists[e2][c_index][3].add((edge1, c_index))
        return CNF_lists

    # TODO AQUÍ CREO QUE ES DONDE SE AÑADEN LOS EDGES EN ORDEN CONTRARIO
    def get_edges_cycle1(self, c):
        edges = []
        for i in range(len(c) - 1):
            edge = [c[i], c[i + 1]]
            edge.sort()
            edges.append(tuple(edge))
        return edges

    def __update_cond_b1(self, G, fundamental_cycles, edge_index_map, 
                        cycle_index_map, CNF_lists):
        # TODO CHEQUEAR BIEN ESTA CONDICIÓN
        for c1, c2 in combinations(fundamental_cycles, 2):
            c1_edges = self.get_edges_cycle1(c1)
            c2_edges = self.get_edges_cycle1(c2)
            c1_not_c2 = [edge for edge in c1_edges if edge not in c2_edges]
            c2_not_c1 = [edge for edge in c2_edges if edge not in c1_edges]
            # print(c1,c1_edges,c2, c2_edges)###
            for edge1 in c1_not_c2:
                for edge2 in c2_not_c1:
                    # print(edge1, edge2)###
                    e1, e2 = edge_index_map[edge1], edge_index_map[edge2]
                    # PN # edge2 respecto C1 ###TODO REVISAR PORQUE AQUÍ TIENE QUE SER CON RESPECTO A DISTINTO CICLO LA QUE SE AÑADE EN LA LISTA PONER CICLOS EN EL RESTO
                    CNF_lists[e1][cycle_index_map[tuple(c2)]][1].add(
                        (edge2, cycle_index_map[tuple(c1)]))
                    # PN # edge1 respecto C2 ###TODO REVISAR PORQUE AQUÍ TIENE QUE SER CON RESPECTO A DISTINTO CICLO LA QUE SE AÑADE EN LA LISTA
                    CNF_lists[e2][cycle_index_map[tuple(c1)]][1].add(
                        (edge1, cycle_index_map[tuple(c2)]))
                    # TODO REVISAR SI METER ÍNDICES DE EDGES
        return CNF_lists

    # Auxiliar functions for condition c)

    def __conflict_type_1_1(self, bridge_pair, c):
        common_att_vert = 0
        for vertex in bridge_pair[0]["att_ver"]:
            if vertex in bridge_pair[1]["att_ver"]:
                common_att_vert += 1
                if common_att_vert >= 3:
                    return True  # TODO REVISAR ESTE BREAK
        return common_att_vert >= 3

    # TODO checkear que los ciclos siempre entran aquí ordenados
    def __conflict_type_2_1(self, bridge_pair, c):
        matching_seq = 0

        # Look for the pattern starting on attachment vertices of both pairs

        # print("#################")
        # print(bridge_pair, c)
        for node in c[0:len(c) - 1]:
            # print()
            # print(node)
            # print(matching_seq)
            if (node in bridge_pair[0]["att_ver"]) and matching_seq % 2 == 0:
                matching_seq += 1
                # print(matching_seq)
                if matching_seq >= 4:
                    return True
            elif (node in bridge_pair[1]["att_ver"]) and matching_seq % 2 == 1:
                matching_seq += 1
                # print(matching_seq)
                if matching_seq >= 4:
                    return True

        # Treat last node of the cycle differently. Otherwise, there can be errors if the starting node
        # of the cycle is of attachment of both bridges.
        if (c[len(c) - 1] in bridge_pair[1]["att_ver"]) and matching_seq % 2 == 1 and (c[len(c) - 1] not in bridge_pair[0]["att_ver"]):
            matching_seq += 1
            # print(matching_seq)
            if matching_seq >= 4:
                return True

        matching_seq = 0
        for node in c[0:len(c) - 1]:
            # print()
            # print(node)
            # print(matching_seq)
            if (node in bridge_pair[1]["att_ver"]) and matching_seq % 2 == 0:
                matching_seq += 1
                # print(matching_seq)
                if matching_seq >= 4:
                    return True
            elif (node in bridge_pair[0]["att_ver"]) and matching_seq % 2 == 1:
                matching_seq += 1
                # print(matching_seq)
                # print(matching_seq >= 4)
                if matching_seq >= 4:
                    return True  # TODO ALGUNOS DE ESTOS SOBRAN SEGÚN LOS MÓDULOS

        # Treat last node of the cycle differently. Otherwise, there can be errors if the starting node
        # of the cycle is of attachment of both bridges.
        if (c[len(c) - 1] in bridge_pair[0]["att_ver"]) and matching_seq % 2 == 1 and (c[len(c) - 1] not in bridge_pair[1]["att_ver"]):
            matching_seq += 1
            if matching_seq >= 4:
                return True

        return matching_seq >= 4

    def __conflict_between1(self, bridge_pair, c):
        # print("conflict:", c, bridge_pair,"---", conflict_type_1(bridge_pair, c), conflict_type_2(bridge_pair, c))
        return self.__conflict_type_1_1(bridge_pair, c) or self.__conflict_type_2_1(bridge_pair, c)

    def __update_cond_c1(
            self, G, fundamental_cycles, bridges_all_cycles, edge_index_map,
            CNF_lists
            ):
        for c_index, c in enumerate(fundamental_cycles):
            for bridge1, bridge2 in combinations(bridges_all_cycles[tuple(c)], 2):
                if self.__conflict_between1((bridge1, bridge2), c):
                    for edge1 in bridge1["edges"]:
                        for edge2 in bridge2["edges"]:
                            e1, e2 = edge_index_map[edge1], edge_index_map[edge2] ### TODO VER SI ES MEJOR METERLO CON INDICES Y AÑADIR CICLOS  AQUÍ
                            # CNF_lists[e1][c_index][1].add(e2)
                            # CNF_lists[e1][c_index][2].add(e2)
                            # CNF_lists[e2][c_index][1].add(e1)
                            # CNF_lists[e2][c_index][2].add(e1)
                            # print("writing conflict between", e1, e2)
                            CNF_lists[e1][c_index][0].add((edge2, c_index))  ### TODO VER SI ES MEJOR METERLO CON INDICES
                            CNF_lists[e2][c_index][0].add((edge1, c_index))
                            CNF_lists[e1][c_index][3].add((edge2, c_index))
                            CNF_lists[e2][c_index][3].add((edge1, c_index))
        return CNF_lists

    def get_2_CNF1(self, G, fundamental_cycles, bridges_all_cycles):
        # Asignamos índices únicos a las aristas del grafo
        # edge_index_map = {edge: i for i, edge in enumerate(G.edges())} ### TODO VER DONDE CAMBIAN DE DIRECCIÓN LOS EDGES Y USAR ESTE INDEX MAP
        # Assign unique indices to edges in the order they appear in G.edges(), w1ith both orientations stored
        edge_index_map = {}
        for i, edge in enumerate(G.edges()):
            edge_index_map[edge] = i
            # Store reversed edge as well
            edge_index_map[(edge[1], edge[0])] = i

        cycle_index_map = {}
        for i, c in enumerate(fundamental_cycles):
            cycle_index_map[tuple(c)] = i

        # Estructura CNF_lists basada en edges
        CNF_lists = [[[set([]) for _ in range(4)] for _ in range(
            len(fundamental_cycles))] for _ in range(len(G.edges()))]
        # 0 PP
        # 1 PN
        # 2 NP
        # 3 NN
        
        ### TODO INCLUIR AQUÍ REFERENCIA DE CONDICIONES A B C EN EL PAPER
        CNF_lists = self.__update_cond_a1(G, fundamental_cycles,
                                       bridges_all_cycles, edge_index_map, 
                                       CNF_lists)
        
        CNF_lists = self.__update_cond_b1(G, fundamental_cycles, edge_index_map, 
                                       cycle_index_map, CNF_lists)
        
        CNF_lists = self.__update_cond_c1(G, fundamental_cycles,
                                       bridges_all_cycles, edge_index_map, 
                                       CNF_lists)
        return CNF_lists
    
    ### FIN FUNCIONES ANTIGUAS DE 2CNF PARA HACERLO CON LISTAS COMO EN EL PAPER CON LA CLASE DEL FINAL DEL ARCHIVO ###        
    #####################################################################################################################   
""" 