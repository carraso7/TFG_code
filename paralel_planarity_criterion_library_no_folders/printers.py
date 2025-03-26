import networkx as nx
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import numpy as np

class Printer:
    
    def print_spanning_tree(self, G, spanning_tree):
        """Plot the original graph and highlight the undirected spanning tree edges."""
        
        pos = nx.spring_layout(G)  # Compute layout for nodes
    
        plt.figure(figsize=(8, 6))
    
        # Draw the original graph edges first (background edges)
        nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.3, width=1)
    
        # Draw spanning tree edges in red
        nx.draw_networkx_edges(spanning_tree, pos, edge_color="red", width=2)
    
        # Draw nodes on top of edges
        nx.draw_networkx_nodes(G, pos, node_color="lightblue", edgecolors="black", node_size=500)
    
        # Draw labels on top
        nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")
    
        plt.title("Graph with Spanning Tree Highlighted")
    
        # Add subtitle below the graph listing the spanning tree edges
        spanning_tree_edges = list(spanning_tree.edges())
        subtitle = f"Spanning Tree Edges: {spanning_tree_edges}"
        plt.figtext(0.5, 0.05, subtitle, wrap=True, horizontalalignment='center', fontsize=10)
        
        plt.show()
        

    def draw_cycle_and_bridges(self, G, bridges_all_cycles, cycle):
        bridges = bridges_all_cycles.get(tuple(cycle), [])
        if not bridges:
            print(f"No bridges found for cycle: {cycle}")
            return

        G_no_c_edges = G.copy()
        G_no_c_edges.remove_edges_from([(cycle[i], cycle[i+1]) for i in range(len(cycle)-1)])

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        pos = nx.spring_layout(G)

        num_bridges = len(bridges)
        colors = cm.rainbow(np.linspace(0, 1, num_bridges))
        colors = [color for color in colors if not np.allclose(color[:3], [0.5, 0.5, 0.5])]

        ### LEFT: Full graph + cycle + colored bridges
        axes[0].set_title("Original Graph with Cycle and Bridges")
        node_colors = ["gray" if node in cycle else "lightblue" for node in G.nodes()]
        nx.draw(G, pos, ax=axes[0], with_labels=True, node_color=node_colors, node_size=700)
        nx.draw_networkx_edges(G, pos, ax=axes[0], edge_color="black")
        nx.draw_networkx_edges(G, pos, edgelist=[(cycle[i], cycle[i+1]) for i in range(len(cycle)-1)], edge_color="gray", ax=axes[0], width=5)
        for bridge, color in zip(bridges, colors):
            nx.draw_networkx_edges(G, pos, edgelist=bridge["edges"], edge_color=[color], ax=axes[0], width=2)

        ### RIGHT: Graph without cycle edges
        axes[1].set_title("Graph without Cycle Edges (Bridges Highlighted)")
        node_colors_no_cycle = ["gray" if node in cycle else "lightgreen" for node in G_no_c_edges.nodes()]
        nx.draw(G_no_c_edges, pos, ax=axes[1], with_labels=True, node_color=node_colors_no_cycle, node_size=700)
        nx.draw_networkx_edges(G_no_c_edges, pos, ax=axes[1], edge_color="black")
        for bridge, color in zip(bridges, colors):
            nx.draw_networkx_edges(G_no_c_edges, pos, edgelist=bridge["edges"], edge_color=[color], ax=axes[1], width=2)

        plt.show()

    
    def print_bridges(self, G, bridges_all_cycles):
        for cycle in bridges_all_cycles:
            self.draw_cycle_and_bridges(G, bridges_all_cycles, list(cycle))
            

    def print_CNF_lists(
            self, CNF_lists, fundamental_cycles, edge_index_map
            ):
        print()
        print("LIST OF IMPLICATION BETWEEN VARIABLES OF THE FORM ((edge_node, edge_node), cycle)")
        for c_index, cycle in enumerate(fundamental_cycles):
            print(f"\nCycle {c_index} (Nodes: {cycle}):")
            for edge, edge_index in edge_index_map.items():
                cnf_conditions = CNF_lists[edge_index][c_index]
                if any(cnf_conditions) and edge[0] < edge[1]:  # Solo imprimir edges que tienen condiciones ###  TODO QUITAR EL AND CUANDO LOS EDGES TENGAN MISMO ORDEN, VIENE DE EDGES DESORDENADOS
                    print(f"  Edge {edge_index}: {edge}")
                    print(f"    PP_c,e (Positive-Positive): {cnf_conditions[0]}")
                    print(f"    PN_c,e (Positive-Negative): {cnf_conditions[1]}")
                    print(f"    NP_c,e (Negative-Positive): {cnf_conditions[2]}")
                    print(f"    NN_c,e (Negative-Negative): {cnf_conditions[3]}")
            print("-" * 50)

