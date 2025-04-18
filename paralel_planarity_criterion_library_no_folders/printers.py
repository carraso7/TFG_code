import networkx as nx
import matplotlib.pyplot as plt
import textwrap

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

### TODO UTILIZAR ESTA CLASE EN LUGAR DE LA DE TRICOMPONENTS EN LOS EJEMPLOS
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
    
