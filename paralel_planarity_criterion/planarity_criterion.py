
import networkx as nx
import random
from itertools import combinations, permutations
import SAT2_solver
import triconnected_components as TCC

class PlanarityCriterion:  

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
        spanning_tree_undirected = nx.Graph(spanning_tree) 

        return spanning_tree_undirected

    
    def fundamental_cycles(self, G, spanning_tree_undirected): 
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
    
    
    def get_bridges(self, G, fundamental_cycles): 
        
        """
        Identifies the bridges of `G` relative to each fundamental cycle 
        
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
            attachment_vertices = [] 
            attachment_edges = [] 
            bridges = []
            
            # Create a copy of the graph and remove cycle edges
            G_no_c_edges = G.copy()
            G_no_c_nodes = G.copy()
            G_no_c_edges.remove_edges_from(
                [(c[i], c[i+1]) for i in range(len(c)-1)])
            G_no_c_nodes.remove_nodes_from(c)
        
            for cycle_node in c[:-1]:  
                for att_edge in G_no_c_edges.edges(cycle_node):
                    # Attachment edge with one node outside of cycle
                    if att_edge[0] not in c or att_edge[1] not in c:
                        attachment_vertices.append(att_edge[0] if att_edge[0] not in c else att_edge[1])
                        # Add edge to attachment edges only if it has one
                        # node outside of the cycle
                        attachment_edges.append(att_edge) 
                    # Attachment edge with both nodes in the cycle (it forms a
                    # bridge on its own).
                    else: 
                        if (att_edge not in attachment_edges) and ((att_edge[1], att_edge[0]) not in attachment_edges):
                            bridge = {
                                "edges": [att_edge],
                                "att_ver": set([att_edge[0], att_edge[1]])
                            }
                            bridges.append(bridge)
                            attachment_vertices.extend([att_edge[0], att_edge[1]])
                            attachment_edges.append(att_edge)
        
            # Eliminate duplicates 
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


    def __get_edges_cycle(self, c):
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


    def __conflict_type_2(self, bridge_pair, c):
        matching_seq = 0

        # Look for the pattern starting on attachment vertices of both pairs
        for node in c[0:len(c) - 1]:
            if (node in bridge_pair[0]["att_ver"]) and matching_seq % 2 == 0:
                matching_seq += 1
                if matching_seq >= 4:
                    return True
            elif (node in bridge_pair[1]["att_ver"]) and matching_seq % 2 == 1:
                matching_seq += 1
                if matching_seq >= 4:
                    return True

        # Treat last node of the cycle differently. Otherwise, there can be errors if the starting node
        # of the cycle is of attachment of both bridges.
        if (c[len(c) - 1] in bridge_pair[1]["att_ver"]) and matching_seq % 2 == 1 and (c[len(c) - 1] not in bridge_pair[0]["att_ver"]):
            matching_seq += 1
            if matching_seq >= 4:
                return True

        matching_seq = 0
        for node in c[0:len(c) - 1]:
            if (node in bridge_pair[1]["att_ver"]) and matching_seq % 2 == 0:
                matching_seq += 1
                if matching_seq >= 4:
                    return True
            elif (node in bridge_pair[0]["att_ver"]) and matching_seq % 2 == 1:
                matching_seq += 1
                if matching_seq >= 4:
                    return True  

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


        Returns
        -------
        results : list of booleans. 
            List of length number of fund. cycles * number of bridges.
            
            Each variable is indexed by a number in range(c * m), where c is
            the number of fundamental cycles and m is the number of edges in 
            `G`. The integer represents the variable with cycle index x and 
            edge index y. Note that the numbers in the range whose edge belongs
            to the cycle do not represent any variable.
        

        """
        
        edge_index_map = {}
        for i, edge in enumerate(G.edges()):
            edge_index_map[edge] = i
            # Store reversed edge as well 
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
        
        for cycle1, cycle2 in permutations(fundamental_cycles, 2): 
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
               
        
    def __sum_cycles(self, edges1, edges2): 
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
    
    def edges_to_cycle(self, edges):  
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
    
    
    def is_planar(self, G): 
        """
        Determines if the graph G is planar using Mac Lane criterion as stated
        in "Paralel planarity criterion"

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
        # If the graph has more than 3*n - 6 edges, return False.
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
            info["failing_tcc"] = "No info"
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
    