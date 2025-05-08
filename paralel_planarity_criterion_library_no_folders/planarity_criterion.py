# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 00:35:41 2025

@author: carlo
"""

import networkx as nx
import random

from collections import deque


from itertools import combinations

import math

import SAT2_solver


class PlanarityCriterion:  # TODO COMENTAR

    def spanning_tree(self, G): ## TODO CAMBIAR A PRIVADO
        # Select a random starting node
        random_node = random.choice(list(G.nodes))

        # Get a spanning tree using DFS
        spanning_tree = nx.dfs_tree(G, source=random_node)

        # Convert spanning tree to an undirected graph
        spanning_tree_undirected = nx.Graph(spanning_tree)

        return spanning_tree_undirected

    # TODO FALTA AQUÍ UN PRINTER QUE HAY QUE INCLUIR EN UN NUEVO ARCHIVO QUE SEA PRINTERS

    # TODO QUITAR LO DE UNDIRECTED
    # TODO VER SI SE PUEDE HACER CON NETWORKX
    def fundamental_cycles(self, G, spanning_tree_undirected): ## TODO CAMBIAR A PRIVADO O CAMBIAR A UNA CLASE DE ÚTILES
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
        # TODO intentar repetir el algoritmo buscando primero componentes triconnexas quitando los edges del ciclo y después haciendo
        # tratamiento especial de aquellas que contienen algún nodo del ciclo, separando esas por el nodo del ciclo. Para ello se puede
        # Hacer DFS desde el nodo del ciclo en las componentes que tengan un ciclo y para cada vecino del nodo del ciclo asignar una nueva
        # componente triconnexa distinta de la del resto de vecinos
        
        ### Ver primero componentes conexas quitando edges del ciclo y luego
        # quitar los nodos del ciclo de esas componentes para ver qué separaciones 
        # provoca el ciclo
        
        """
        Separar 3 tipos de edges, según cuántos nodos del ciclo tienen. Hacer
        componentes conexas del grafo sin el ciclo (quitando nodos y edges) y 
        luego ir edge por edge de los que tienen algún nodo en el ciclo 
        clasificándolos.
        """

        bridges_all_cycles = {}
        attachment_vertices_all_cycles = {}

        
        for c in fundamental_cycles:
            #print("---GETTING BRIDGES---")  ######TODO
            #print("cycle", c)
            attachment_vertices = [] ### TODO CREO QUE SE PUEDE HACER SIN ATT VERT GLOBAL
            attachment_edges = [] ### TODO CHEKEAR SI ESTA LISTA VIOLA PARALELIZACIÓN.
            bridges = []
            
            # Create a copy of the graph and remove cycle edges
            G_no_c_edges = G.copy()
            G_no_c_nodes = G.copy()
            G_no_c_edges.remove_edges_from(
                [(c[i], c[i+1]) for i in range(len(c)-1)])
            G_no_c_nodes.remove_nodes_from(c)
        
            for cycle_node in c[:-1]:  ### TODO REVISAR REPETICIONES AQUÍ.
                #print(cycle_node, "edges: ", G_no_c_edges.edges(cycle_node)) ######TODO
                for att_edge in G_no_c_edges.edges(cycle_node):
                    #print("att_edge:", att_edge) ######TODO
                    if att_edge[0] not in c or att_edge[1] not in c:
                        attachment_vertices.append(att_edge[0] if att_edge[0] not in c else att_edge[1])
                        # Add edge to attachment edges only if it has one
                        # node outside of the cycle
                        attachment_edges.append(att_edge) 
                    else: ### TODO INVERT THIS IF AND MAKE ELIF WITH NEXT CLAUSE
                        if (att_edge not in attachment_edges) and ((att_edge[1], att_edge[0]) not in attachment_edges):
                            #print(attachment_edges) ### TODO QUITAR Y ABAJO TMBN
                            #print((att_edge not in attachment_edges) , (att_edge[1], att_edge[0] not in attachment_edges))
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
        
            # Obtener componentes conectados
            connected_comps = list(nx.connected_components(G_no_c_nodes))
        
            # Añadir los enlaces de attachment a las componentes
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
            
            ### TODO AQUÍ FALTA  LO DE PRINTAR CICLOS y bridges

        return bridges_all_cycles
        # print(attachment_vertices_all_cycles)  ### ELIMINADO E INTRODUCIDO EN DICT

    """    
    def get_bridges1(self, G, fundamental_cycles):
        # TODO intentar repetir el algoritmo buscando primero componentes triconnexas quitando los edges del ciclo y después haciendo
        # tratamiento especial de aquellas que contienen algún nodo del ciclo, separando esas por el nodo del ciclo. Para ello se puede
        # Hacer DFS desde el nodo del ciclo en las componentes que tengan un ciclo y para cada vecino del nodo del ciclo asignar una nueva
        # componente triconnexa distinta de la del resto de vecinos
        
        ### Ver primero componentes conexas quitando edges del ciclo y luego
        # quitar los nodos del ciclo de esas componentes para ver qué separaciones 
        # provoca el ciclo

        bridges_all_cycles = {}
        attachment_vertices_all_cycles = {}

        for c in fundamental_cycles:

            # Create a copy of the graph and remove cycle edges
            G_no_c_edges = G.copy()
            G_no_c_nodes = G.copy()
            G_no_c_edges.remove_edges_from(
                [(c[i], c[i+1]) for i in range(len(c)-1)])
            G_no_c_nodes.remove_nodes_from(c)

            # habría que poner los del ciclo?? para empezar no pero como visited?
            unvisited = list(G_no_c_edges.edges())
            bridges = []
            attachment_vertices = []  # TODO SE PUEDE PRESCINDIR DE ESTA VARIABLE
            q = deque()
            # print("unvisited", unvisited)
            for edge1 in G_no_c_edges.edges():
                if edge1 in unvisited:  # TODO VER SI SE PUEDE MODIFICAR LA LISTA DINÁMICAMENTE, FORMANDO UN SOLO FOR AQUÍ EN VEZ DE IF Y FOR
                    # print("edge", edge1)
                    q.append(edge1)
                    unvisited.remove(edge1)
                    bridge = {"edges": [], "att_ver": set([])}
                    bridge["edges"].append(edge1)
                    # attachment_vertices_bridge = set([]) ### ELIMINADO E INTRODUCIDO EN DICT
                    while q:
                        edge2 = q.popleft()  # Current edge being processed

                        for node in edge2:
                            if node in c:
                                # attachment_vertices_bridge.add(node) ### ELIMINADO E INTRODUCIDO EN DICT
                                bridge["att_ver"].add(node)
                            else:
                                for neighbor in G_no_c_edges.neighbors(node):
                                    # print(v, "vecino:" ,neighbor)
                                    new_edge = (node, neighbor)
                                    new_edge_rev = (neighbor, node)
                                    # print(new_edge)
                                    # print(unvisited)
                                    # Ensure it's unvisited and not in the cycle
                                    if (new_edge in unvisited) or (new_edge_rev in unvisited):
                                        # print("añadir")
                                        if new_edge in unvisited:
                                            unvisited.remove(new_edge)
                                        else:
                                            unvisited.remove(new_edge_rev)
                                        if neighbor in c:
                                            # attachment_vertices_bridge.add(neighbor) ######
                                            bridge["att_ver"].add(neighbor)
                                        else:
                                            q.append(new_edge)
                                        bridge["edges"].append(new_edge)
                    # Store the final edge-connected component
                    bridges.append(bridge)
                    # attachment_vertices.append(attachment_vertices_bridge)  ### ELIMINADO E INTRODUCIDO EN DICT
            bridges_all_cycles[tuple(c)] = bridges
            # attachment_vertices_all_cycles[tuple(c)] = attachment_vertices  ### ELIMINADO E INTRODUCIDO EN DICT

            # TODO AQUÍ FALTA  LO DE PRINTAR CICLOS y bridges

        return bridges_all_cycles
        # print(attachment_vertices_all_cycles)  ### ELIMINADO E INTRODUCIDO EN DICT
    """
    
    ### Auxiliar functions for getting 2 CNF conditions ###
        

    def __update_cond_a(
            self, G, fundamental_cycles, bridges_all_cycles, edge_index_map, 
            implications
            ):
        n_edges = len(edge_index_map)
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
                    implications.append((offset + edge1, offset + edge2))
                    # e2 -> e1
                    implications.append((offset + edge2, offset + edge1))
                    # ¬e1 -> ¬e2
                    implications.append((neg_offset + edge1, neg_offset + edge2))
                    # ¬e2 -> ¬e1
                    implications.append((neg_offset + edge2, neg_offset + edge1))
                    
        return implications

    ### TODO SEGUIR DESDE AQUÍ
    # TODO AQUÍ CREO QUE ES DONDE SE AÑADEN LOS EDGES EN ORDEN CONTRARIO
    def __get_edges_cycle(self, c):
        edges = []
        for i in range(len(c) - 1):
            edge = [c[i], c[i + 1]]
            edge.sort()
            edges.append(tuple(edge))
        return edges

    def __update_cond_b(self, G, fundamental_cycles, edge_index_map, 
                        cycle_index_map, implications):
        n_edges = len(edge_index_map)
        n_cycles = len(fundamental_cycles)
        neg_offset = n_edges * n_cycles
        # TODO CHEQUEAR BIEN ESTA CONDICIÓN
        for cycle1, cycle2 in combinations(fundamental_cycles, 2):
            c1_edges = self.__get_edges_cycle(cycle1)
            c2_edges = self.__get_edges_cycle(cycle2)
            c1 = cycle_index_map[cycle1]
            c2 = cycle_index_map[cycle2]
            c1_not_c2 = [edge for edge in c1_edges if edge not in c2_edges]
            c2_not_c1 = [edge for edge in c2_edges if edge not in c1_edges]
            # print(c1,c1_edges,c2, c2_edges)###
            for edge1 in c1_not_c2:
                for edge2 in c2_not_c1:
                    # e1c2 -> ¬e2c1
                    implications.append((c2*n_edges + edge1, neg_offset + c1*n_edges + edge2))
                    # e2c1 -> ¬e1c2
                    implications.append((c1*n_edges + edge2, neg_offset + c2*n_edges + edge1))
        return implications

    # Auxiliar functions for condition c)

    def __conflict_type_1(self, bridge_pair, c):
        common_att_vert = 0
        for vertex in bridge_pair[0]["att_ver"]:
            if vertex in bridge_pair[1]["att_ver"]:
                common_att_vert += 1
                if common_att_vert >= 3:
                    return True  # TODO REVISAR ESTE BREAK
        return common_att_vert >= 3

    # TODO checkear que los ciclos siempre entran aquí ordenados
    def __conflict_type_2(self, bridge_pair, c):
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

    def __conflict_between(self, bridge_pair, c):
        # print("conflict:", c, bridge_pair,"---", conflict_type_1(bridge_pair, c), conflict_type_2(bridge_pair, c))
        return self.__conflict_type_1(bridge_pair, c) or self.__conflict_type_2(bridge_pair, c)

    def __update_cond_c(
            self, G, fundamental_cycles, bridges_all_cycles, edge_index_map,
            implications
            ):
        n_edges = len(edge_index_map)
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
        
        implications = []
        
        implications = self.__update_cond_a(
            G, fundamental_cycles, bridges_all_cycles, edge_index_map, implications
            )
        
        implications = self.update_cond_b(
            G, fundamental_cycles, edge_index_map, cycle_index_map, implications
            )
        
        implications = self.__update_cond_c(
            G, fundamental_cycles, bridges_all_cycles, edge_index_map, implications
            )
        
        return implications
    
    
    def get_truth_assigment(self, G, fundamental_cycles, bridges_all_cycles):
        
        edge_index_map = {}
        for i, edge in enumerate(G.edges()):
            edge_index_map[edge] = i
            # Store reversed edge as well ### TODO INTENTAR NO USAR ESTAS LÍNEAS
            # edge_index_map[(edge[1], edge[0])] = i
            
        cycle_index_map = {}
        for i, c in enumerate(fundamental_cycles):
            cycle_index_map[tuple(c)] = i
                
        ### TODO COMENTAR COMO FUNCIONA IMPLICATIONS
        # num_ciclo*num_edges + num_edge es variable positiva
        # num_ciclos*num_edges + num_ciclo*num_edges + num edge es variable negativa
        implications = self.__get_implications_2CNF(
            G, fundamental_cycles, bridges_all_cycles, edge_index_map, 
            cycle_index_map
            )
        solver = SAT2_solver.SAT2_solver()
        n_variables = len(fundamental_cycles) * len(G.edges())
        results, info = solver.get_truth_assigment(implications, n_variables)
        info["edge_index_map"] = edge_index_map
        info["cycle_index_map"] = cycle_index_map
        return results, info 
        ## TODO IMPORT AND CALL 2CNF SOLVER
                
        
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
    
    def compute_lt(self, truth_assign, fundamental_cycles, info):
        rel_lt = [[0 for _ in range(fundamental_cycles)] for _ in range(fundamental_cycles)]
        rel_in = [[0 for _ in range(fundamental_cycles)] for _ in range(fundamental_cycles)]
        cycle_index_map = info["cycle_index_map"]
        edge_index_map = info["edge_index_map"]
        
        for cycle1, cycle2 in combinations(fundamental_cycles, 2):
            c1_edges = self.__get_edges_cycle(cycle1)
            c2_edges = self.__get_edges_cycle(cycle2)
            c1 = cycle_index_map[cycle1]
            c2 = cycle_index_map[cycle2]
            for edge in c1_edges:
                if edge not in c2_edges:    
                    c1_not_c2 = edge_index_map[edge]
                    break
            for edge in c2_edges:
                if edge not in c1_edges:
                    c2_not_c1 = edge_index_map[edge]
                    break
            if truth_assign[c2*len(edge_index_map) + c1_not_c2]:
                rel_in[c1][c2] = 1
            if truth_assign[c1*len(edge_index_map) + c2_not_c1]:
                rel_in[c2][c1] = 1
                
        for cycle1, cycle2 in combinations(fundamental_cycles, 2):
            c1_edges = self.__get_edges_cycle(cycle1)
            c2_edges = self.__get_edges_cycle(cycle2)
            c1 = cycle_index_map[cycle1]
            c2 = cycle_index_map[cycle2]
            if rel_lt[c1][c2]:
                rel_lt[c1][c2] = 1
                for cycle3 in fundamental_cycles:    
                    c3 = cycle_index_map[cycle3]
                    if rel_lt[c1][c3] and rel_lt[c3][c2]:
                        rel_lt[c1][c2] = 0
                        break
                    if rel_lt[c1][c2] == 0:
                        break
                    
        info["rel_in"] = rel_in
        return rel_lt, info
               
        
    
    
    def get_peripheral_basis(self, ):
        pass
    
    
    def get_plane_mesh(self,):
        pass
    
    
    def is_planar(self,):
        pass

### TODO HAY AQUÍ UN PRINTER QUE FALTA POR PONER de cnf_lists_by_cycle

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