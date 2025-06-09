import networkx as nx
import matplotlib.pyplot as plt
import textwrap

import matplotlib.cm as cm
import numpy as np

import os

class Printer:
    
    def print_spanning_tree(self, G, spanning_tree, save=False, 
                            name="spanning tree", dir_name="images"
                            ):
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
        
        if save:
            os.makedirs(dir_name, exist_ok=True)
    
            # Set the save path (always PNG)
            save_path = os.path.join('images', f'{name}.png')
            plt.savefig(save_path, bbox_inches='tight')
        
        plt.show()
        

    def draw_cycle_and_bridges(self, G, bridges_all_cycles, cycle, save=False, 
                            name="cycle and bridges", dir_name="images"):
        """
        Draws a cycle of a graph with all its bridges with different colors.

        Parameters ### TODO ACABAR DE ESCRIBIR O QUITAR.
        ----------
        G : TYPE
            DESCRIPTION.
        bridges_all_cycles : TYPE
            DESCRIPTION.
        cycle : TYPE
            DESCRIPTION.
        save : TYPE, optional
            DESCRIPTION. The default is False.
        name : TYPE, optional
            DESCRIPTION. The default is "cycle and bridges".
        dir_name : TYPE, optional
            DESCRIPTION. The default is "images".

        Returns
        -------
        None.

        """
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
        
        if save:
            os.makedirs(dir_name, exist_ok=True)
    
            # Set the save path (always PNG)
            save_path = os.path.join('images', f'{name}.png')
            plt.savefig(save_path, bbox_inches='tight')
            
        plt.show()

    
    def print_bridges(self, G, bridges_all_cycles, save=False, 
                            name="cycle and bridges", dir_name="images"):
        """
        Prints all the bridges of each cycle with different colors.

        Parameters ### TODO ACABAR DE ESCRIBIR O QUITAR.
        ----------
        G : TYPE
            DESCRIPTION.
        bridges_all_cycles : TYPE
            DESCRIPTION.
        save : TYPE, optional
            DESCRIPTION. The default is False.
        name : TYPE, optional
            DESCRIPTION. The default is "cycle and bridges".
        dir_name : TYPE, optional
            DESCRIPTION. The default is "images".

        Returns
        -------
        None.

        """
        for i, cycle in enumerate(bridges_all_cycles):
            self.draw_cycle_and_bridges(G, bridges_all_cycles, list(cycle),
                                        save=save, name=(name +  str(i)), 
                                        dir_name=dir_name
                                        )
            

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
            
    def print_cycle_edge_table(boolean_list, edge_map, cycle_map):
        """
        Prints a truth assigment as a table of boolean variables.

        Parameters ### TODO ACABAR DE ESCRIBIR O QUITAR.
        ----------
        boolean_list : TYPE
            DESCRIPTION.
        edge_map : TYPE
            DESCRIPTION.
        cycle_map : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # Get the number of edges (columns) and cycles (rows)
        num_edges = len(set(edge_map.values()))
        num_cycles = len(cycle_map)
    
        # Invert cycle_map to get row index to cycle mapping
        cycle_indices = {v: k for k, v in cycle_map.items()}
        edge_indices = {v: k for k, v in edge_map.items() if v in set(edge_map.values())}
    
        # Prepare column widths based on edge tuple lengths
        col_widths = []
        for col in range(num_edges):
            edge_label = str(edge_indices[col])
            col_widths.append(len(edge_label) + 2)  # +2 for padding
    
        # Determine max width for the first column based on the longest cycle label
        max_cycle_len = max(len(str(cycle_indices[row])) for row in range(num_cycles))
        first_col_width = max(max_cycle_len, len("Cycle/Edge")) + 2  # Add padding
    
        # Print header
        header = "Cycle/Edge".ljust(first_col_width)
        for col in range(num_edges):
            edge_label = str(edge_indices[col])
            header += edge_label.ljust(col_widths[col])
        print(header)
        print('-' * len(header))
    
        # Print each row
        for row in range(num_cycles):
            cycle = str(cycle_indices[row])
            if (boolean_list): #### TODO VER CASOS EN LOS QUE NO EXISTE
                row_values = boolean_list[row * num_edges: (row + 1) * num_edges]
                row_str = cycle.ljust(first_col_width)
                for col, val in enumerate(row_values):
                    row_str += ('True' if val else 'False').ljust(col_widths[col])
                print(row_str)
            
    def print_B_matrix(matrix, name):
        n = len(matrix)
        print(f"\n{name}:")
    
        # Print column headers
        header = "   " + ' '.join(f"{j:2}" for j in range(n))
        print(header)
    
        # Print each row with its index
        for i, row in enumerate(matrix):
            row_str = f"{i:2}|" + ' '.join(f"{' -' if elem == float('-inf') else f'{elem:2}'}" for elem in row)
            print(row_str)
    def print_B_matrix1(matrix, name):
        n = len(matrix)
        print(f"\n{name}:")

        # Print column headers
        header = "   " + ' '.join(f"{j:2}" for j in range(n))
        print(header)

        # Print each row with its index
        for i, row in enumerate(matrix):
            row_str = f"{i:2}|" + ' '.join(f"{'-' if elem == float('-inf') else f'{elem:2}'}" for elem in row)
            print(row_str)

### TODO UTILIZAR ESTA CLASE EN LUGAR DE LA DE TRICOMPONENTS EN LOS EJEMPLOS
class ConnectedComponentsDrawer(): ### TODO LEER SI EN PYTHON ES CORRECTO PONER MÁS DE UNA CLASE EN EL MISMO ARCHIVO O DEBEN ESTAR EN ARCHIVOS SEPARADOS. 
### TODO ESCRIBIR A NETWORKX PARA VER SI QUIEREN MI LIBRERÍA

    def print_n_connected_components(self, G, NCC, max_line_length=40, N=-1,
                                     save=False, name="connected components", 
                                     dir_name="images"
                                     ):
        """
        Prints the N connected components of the graph. In the case N=3, it 
        prints the components as the definition in the paper, also called
        triconnected blocks in this case.

        Parameters ### TODO ACABAR DE ESCRIBIR O QUITAR.
        ----------
        G : TYPE
            DESCRIPTION.
        NCC : TYPE
            DESCRIPTION.
        max_line_length : TYPE, optional
            DESCRIPTION. The default is 40.
        N : TYPE, optional
            DESCRIPTION. The default is -1.
        save : TYPE, optional
            DESCRIPTION. The default is False.
        name : TYPE, optional
            DESCRIPTION. The default is "connected components".
        dir_name : TYPE, optional
            DESCRIPTION. The default is "images".

        Returns
        -------
        None.

        """
        import matplotlib.pyplot as plt
        import networkx as nx
        import textwrap
    
        # Initialize classification dictionaries
        node_classes = {}
        edge_classes = {}
        virtual_edge_classes = {} 
    
        # Assign nodes and edges to their NCCs
        for class_idx, component in enumerate(NCC):
            if N == 3:
                nodes = component["node_list"]
                virt_edges = component["virtual_edges"]
            else:
                nodes = component
                virt_edges = []
    
            for node in nodes:
                node_classes.setdefault(node, []).append(class_idx)
    
            for edge in G.edges():
                if edge[0] in nodes and edge[1] in nodes:
                    edge_classes.setdefault(edge, []).append(class_idx)
    
            # Collect virtual edges info (only for N==3)
            for v_edge in virt_edges:
                # Normalize edge to (small, large) to handle undirected graphs
                v_edge_norm = tuple(sorted(v_edge))
                virtual_edge_classes.setdefault(v_edge_norm, []).append(class_idx)
    
        # Define colors:
        num_components = len(NCC)
        color_map = plt.cm.get_cmap("tab10", num_components)
    
        node_colors = []
        node_labels = {}
        for node in G.nodes():
            if node not in node_classes:
                node_colors.append("black")
                node_labels[node] = "white"
            elif len(node_classes[node]) > 1:
                node_colors.append("gray")
                node_labels[node] = "black"
            else:
                node_colors.append(color_map(node_classes[node][0]))
                node_labels[node] = "black"
    
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
    
        nx.draw(
            G, pos, with_labels=False, node_color=node_colors, edge_color=edge_colors,
            node_size=600, edge_cmap=color_map, font_size=10, ax=ax
        )
    
        # Draw labels with appropriate font color
        for node, (x, y) in pos.items():
            plt.text(x, y, str(node), ha='center', va='center', color=node_labels[node])
    
        # Draw virtual edges (only for N==3)
        if N == 3:
            for v_edge, classes in virtual_edge_classes.items():
                x0, y0 = pos[v_edge[0]]
                x1, y1 = pos[v_edge[1]]
                color = "gray" if len(classes) > 1 else color_map(classes[0])
                ax.plot(
                    [x0, x1], [y0, y1],
                    linestyle='--',
                    color=color,
                    linewidth=2
                )
    
        N_label = str(N) if N != -1 else "N"
    
        plt.title(f"Graph with {N_label}-connected Components\n(Gray = Shared, Black = Uncategorized)")
    
        # Adjust margins for text
        plt.subplots_adjust(left=0.3)
    
        # Format the NCC list with line wrapping
        formatted_ncc_list = []
        for i, component in enumerate(NCC):
            if N == 3:
                comp_nodes = sorted(list(component["node_list"]))
                comp_str = f"{N_label}CC {i+1}: {comp_nodes}"
            else:
                comp_str = f"{N_label}CC {i+1}: {sorted(list(component))}"
            wrapped_lines = textwrap.wrap(comp_str, width=max_line_length)
            formatted_ncc_list.extend(wrapped_lines)
    
        subtitle_text = "\n".join(formatted_ncc_list)
        plt.figtext(
            0.02, 0.5, subtitle_text, wrap=True,
            horizontalalignment='left', verticalalignment='center', fontsize=10
        )
        
        if save:
            os.makedirs(dir_name, exist_ok=True)
    
            # Set the save path (always PNG)
            save_path = os.path.join('images', f'{name}.png')
            plt.savefig(save_path, bbox_inches='tight')
    
        plt.show()
        
    ### IGUAL QUE EL ANTERIOR PERO ESPECIFICANDO POSICIONES    
    def print_n_connected_components_fixed_positions(self, G, NCC, max_line_length=40, N=-1,
                                 save=False, name="connected components", 
                                 dir_name="images", fixed_pos=None):
        """
        Prints the N connected components of the graph. In the case N=3, it 
        prints the components as the definition in the paper, also called
        triconnected blocks in this case.
    
        Parameters
        ----------
        G : networkx.Graph
            The full graph to analyze and plot.
        NCC : list
            List of connected components or triconnected blocks (dicts for N=3).
        max_line_length : int
            Maximum characters per line for labeling component lists.
        N : int
            The connectivity level (e.g., 2, 3). Use 3 for triconnected blocks.
        save : bool
            If True, saves the figure to the given directory.
        name : str
            Filename to use when saving (without extension).
        dir_name : str
            Folder where the image is saved if save=True.
        fixed_pos : dict, optional
            Dictionary like {node: (x, y)} to fix positions of selected nodes
            (e.g., {4: (0, 0)}).
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        import textwrap
    
        fixed_pos = fixed_pos or {}
    
        # Initialize classification dictionaries
        node_classes = {}
        edge_classes = {}
        virtual_edge_classes = {}
    
        for class_idx, component in enumerate(NCC):
            nodes = component["node_list"] if N == 3 else component
            virt_edges = component["virtual_edges"] if N == 3 else []
    
            for node in nodes:
                node_classes.setdefault(node, []).append(class_idx)
            for edge in G.edges():
                if edge[0] in nodes and edge[1] in nodes:
                    edge_classes.setdefault(edge, []).append(class_idx)
            for v_edge in virt_edges:
                v_edge_norm = tuple(sorted(v_edge))
                virtual_edge_classes.setdefault(v_edge_norm, []).append(class_idx)
    
        num_components = len(NCC)
        color_map = plt.cm.get_cmap("tab10", num_components)
    
        node_colors = []
        node_labels = {}
        for node in G.nodes():
            if node not in node_classes:
                node_colors.append("black")
                node_labels[node] = "white"
            elif len(node_classes[node]) > 1:
                node_colors.append("gray")
                node_labels[node] = "black"
            else:
                node_colors.append(color_map(node_classes[node][0]))
                node_labels[node] = "black"
    
        edge_colors = []
        for edge in G.edges():
            if edge not in edge_classes:
                edge_colors.append("black")
            elif len(edge_classes[edge]) > 1:
                edge_colors.append("gray")
            else:
                edge_colors.append(color_map(edge_classes[edge][0]))
    
        fig, ax = plt.subplots(figsize=(7, 5))
    
        # Compute layout
        pos = nx.spring_layout(
            G,
            seed=42,
            pos=fixed_pos,
            fixed=fixed_pos.keys()
        )
    
        nx.draw(
            G, pos, with_labels=False, node_color=node_colors, edge_color=edge_colors,
            node_size=600, edge_cmap=color_map, font_size=10, ax=ax
        )
    
        for node, (x, y) in pos.items():
            plt.text(x, y, str(node), ha='center', va='center', color=node_labels[node])
    
        if N == 3:
            for v_edge, classes in virtual_edge_classes.items():
                x0, y0 = pos[v_edge[0]]
                x1, y1 = pos[v_edge[1]]
                color = "gray" if len(classes) > 1 else color_map(classes[0])
                ax.plot([x0, x1], [y0, y1], linestyle='--', color=color, linewidth=2)
    
        N_label = str(N) if N != -1 else "N"
        plt.title(f"Graph with {N_label}-connected Components\n(Gray = Shared, Black = Uncategorized)")
        plt.subplots_adjust(left=0.3)
    
        formatted_ncc_list = []
        for i, component in enumerate(NCC):
            comp_nodes = sorted(component["node_list"]) if N == 3 else sorted(component)
            comp_str = f"{N_label}CC {i+1}: {comp_nodes}"
            formatted_ncc_list.extend(textwrap.wrap(comp_str, width=max_line_length))
    
        plt.figtext(0.02, 0.5, "\n".join(formatted_ncc_list), wrap=True,
                    horizontalalignment='left', verticalalignment='center', fontsize=10)
    
        if save:
            os.makedirs(dir_name, exist_ok=True)
            save_path = os.path.join(dir_name, f'{name}.png')
            plt.savefig(save_path, bbox_inches='tight')
    
        plt.show()


        
class SPQR_drawer():

    @staticmethod
    def plot_SPQR_visuals(G, edges_dict, SPQR_tree, save=False,
                          name="SPQR_tree", dir_name="images"
                          ):
        """
        Plots the SPQR tree and the full graph with SPQR tree edges highlighted.
    
        Parameters
        ----------
        G : networkx.Graph
            The original graph.
        TCCs : list of dicts
            Triconnected components (output from triconnected_comps).
        """
        real_edges = set(edges_dict["real_edges"])
        virtual_edges = set(edges_dict["virtual_edges"])
    
        pos = nx.spring_layout(G, seed=42)  # Same layout for both plots
    
        # === 1. Plot SPQR tree ===
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("SPQR Tree")
    
        nx.draw_networkx_nodes(SPQR_tree, pos, node_color='lightblue', edgecolors='k')
        nx.draw_networkx_labels(SPQR_tree, pos)
    
        # Draw real edges (solid grey)
        nx.draw_networkx_edges(SPQR_tree, pos, edgelist=real_edges, edge_color='grey', style='solid', width=2)
        # Draw virtual edges (dashed grey)
        nx.draw_networkx_edges(SPQR_tree, pos, edgelist=virtual_edges, edge_color='grey', style='dashed', width=2)
    
        plt.axis('off')
    
        # === 2. Plot original graph with SPQR tree edges styled ===
        plt.subplot(1, 2, 2)
        plt.title("Graph G with SPQR Tree Edges Highlighted")
    
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', edgecolors='k')
        nx.draw_networkx_labels(G, pos)
    
        # Non-tree edges (black)
        other_edges = [e for e in G.edges() if (e not in real_edges and tuple(reversed(e)) not in real_edges
                                                and e not in virtual_edges and tuple(reversed(e)) not in virtual_edges)]
        nx.draw_networkx_edges(G, pos, edgelist=other_edges, edge_color='grey', style='solid', width=1.5)
    
        # Tree real edges (solid grey)
        nx.draw_networkx_edges(G, pos, edgelist=real_edges, edge_color='black', style='solid', width=2)
    
        # Tree virtual edges (dashed grey)
        nx.draw_networkx_edges(G, pos, edgelist=virtual_edges, edge_color='black', style='dashed', width=2)
    
        plt.axis('off')
        plt.tight_layout()
        
        if save:
            os.makedirs(dir_name, exist_ok=True)
    
            # Set the save path (always PNG)
            save_path = os.path.join(dir_name, f'{name}.png')
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Image saved to: {save_path}")
        
        plt.show()
        
    @staticmethod
    def plot_SPQR_and_TCC(G, TCCs, edges_dict, SPQR_tree, save=False,
                          name="SPQR_and_TCC", dir_name="images"
                          ):
        import matplotlib.pyplot as plt
        import networkx as nx
        import os
        import textwrap
    
        real_edges = set(tuple(sorted(e)) for e in edges_dict["real_edges"])
        virtual_edges = set(tuple(sorted(e)) for e in edges_dict["virtual_edges"])
    
        # Maps from nodes/edges to the TCCs they belong to
        node_classes = {}
        edge_classes = {}
        virtual_edge_classes = {}
    
        for idx, tcc in enumerate(TCCs):
            nodes = tcc["node_list"]
            virt_edges = tcc["virtual_edges"]
    
            for node in nodes:
                node_classes.setdefault(node, []).append(idx)
    
            for edge in G.edges():
                if edge[0] in nodes and edge[1] in nodes:
                    edge_norm = tuple(sorted(edge))
                    edge_classes.setdefault(edge_norm, []).append(idx)
    
            for v_edge in virt_edges:
                v_edge_norm = tuple(sorted(v_edge))
                virtual_edge_classes.setdefault(v_edge_norm, []).append(idx)
    
        num_components = len(TCCs)
        color_map = plt.cm.get_cmap("tab10", num_components)
    
        # Identify all nodes involved in SPQR tree edges
        tree_nodes = {n for edge in real_edges.union(virtual_edges) for n in edge}
    
        # Assign node colors and label colors
        node_colors = []
        label_colors = {}
        for node in G.nodes():
            if node in tree_nodes:
                node_colors.append("black")
                label_colors[node] = "white"
            elif node not in node_classes:
                node_colors.append("lightgray")
                label_colors[node] = "black"
            elif len(node_classes[node]) > 1:
                node_colors.append("gray")
                label_colors[node] = "black"
            else:
                node_colors.append(color_map(node_classes[node][0]))
                label_colors[node] = "black"
    
        # Assign edge colors and styles
        edge_colors = []
        edge_widths = []
        edge_styles = []
    
        for edge in G.edges():
            e = tuple(sorted(edge))
            if e in real_edges:
                edge_colors.append("black")
                edge_widths.append(3)
                edge_styles.append("solid")
            elif e in virtual_edges:
                edge_colors.append("black")
                edge_widths.append(3)
                edge_styles.append("dashed")
            elif e not in edge_classes:
                edge_colors.append("lightgray")
                edge_widths.append(1.5)
                edge_styles.append("solid")
            elif len(edge_classes[e]) > 1:
                edge_colors.append("gray")
                edge_widths.append(1.5)
                edge_styles.append("solid")
            else:
                edge_colors.append(color_map(edge_classes[e][0]))
                edge_widths.append(1.5)
                edge_styles.append("solid")
    
        # --- Drawing ---
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)
    
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, edgecolors='k', node_size=600)
    
        # Draw node labels
        for node, (x, y) in pos.items():
            plt.text(x, y, str(node), ha='center', va='center', color=label_colors[node])
    
        # Draw edges
        for (edge, color, width, style) in zip(G.edges(), edge_colors, edge_widths, edge_styles):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            ax.plot([x0, x1], [y0, y1], color=color, linewidth=width, linestyle=style, zorder=0)
    
        # Draw virtual edges (not in G but referenced in TCCs)
        for v_edge, classes in virtual_edge_classes.items():
            if not G.has_edge(*v_edge):
                x0, y0 = pos[v_edge[0]]
                x1, y1 = pos[v_edge[1]]
                color = "black" if v_edge in virtual_edges else (
                    "gray" if len(classes) > 1 else color_map(classes[0]))
                style = "dashed"
                width = 3 if v_edge in virtual_edges else 1.5
                ax.plot([x0, x1], [y0, y1], linestyle=style, color=color, linewidth=width)
    
        plt.title("Graph with SPQR Tree (Black) and Triconnected Components")
        plt.axis('off')
    
        # Sidebar list of TCCs
        formatted_ncc_list = []
        for i, component in enumerate(TCCs):
            comp_nodes = sorted(list(component["node_list"]))
            comp_str = f"TCC {i+1}: {comp_nodes}"
            wrapped_lines = textwrap.wrap(comp_str, width=40)
            formatted_ncc_list.extend(wrapped_lines)
    
        subtitle_text = "\n".join(formatted_ncc_list)
        plt.figtext(0.02, 0.5, subtitle_text, wrap=True, horizontalalignment='left',
                    verticalalignment='center', fontsize=10)
    
        if save:
            os.makedirs(dir_name, exist_ok=True)
            path = os.path.join(dir_name, f"{name}.png")
            plt.savefig(path, bbox_inches="tight")
    
        plt.show()


