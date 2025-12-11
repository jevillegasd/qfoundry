"""
Topology Design Module for QFoundry

This module provides tools for generating, analyzing, and coloring quantum processor topologies.
It wraps RustworkX generators and provides custom coloring algorithms for frequency allocation.

Author: QFoundry Design Team
Date: December 2025
"""

import numpy as np
import rustworkx as rx
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from rustworkx import ColoringStrategy

@dataclass
class LatticeInfo:
    """Information about a generated lattice."""
    lattice_type: str
    num_nodes: int
    num_edges: int
    shape: Tuple[int, int]
    node_positions: Dict[int, Tuple[float, float]]
    parameters: Dict


class LatticeGenerator:
    """Clean wrapper around RustworkX lattice generators."""
    
    def __init__(self):
        self.last_generated_info: Optional[LatticeInfo] = None
    
    def generate_square_lattice(self, rows: int, cols: int) -> Tuple[rx.PyGraph, LatticeInfo]:
        """
        Generate a square lattice using RustworkX.
        
        Args:
            rows: Number of rows in the lattice
            cols: Number of columns in the lattice
        
        Returns:
            Tuple of (graph, lattice_info)
        """

        graph = rx.generators.grid_graph(rows, cols)
        
        # Calculate node positions for visualization
        node_positions = {}
        for node_idx in graph.node_indices():
            # Convert linear index back to (row, col) coordinates
            row = node_idx // cols
            col = node_idx % cols
            node_positions[node_idx] = (float(col), float(row))
        
        info = LatticeInfo(
            lattice_type="square",
            num_nodes=graph.num_nodes(),
            num_edges=graph.num_edges(),
            shape=(rows, cols),
            node_positions=node_positions,
            parameters={"periodic": False}
        )
        
        self.last_generated_info = info
        return graph, info
    
    def generate_hexagonal_lattice(self, rows: int, cols: int, periodic: bool = False) -> Tuple[rx.PyGraph, LatticeInfo]:
        """
        Generate a hexagonal lattice using RustworkX.
        
        Args:
            rows: Number of rows in the lattice
            cols: Number of columns in the lattice  
            periodic: Whether to use periodic boundary conditions
        
        Returns:
            Tuple of (graph, lattice_info)
        """
        # Use RustworkX hexagonal lattice generator
        graph = rx.generators.hexagonal_lattice_graph(rows, cols, periodic=periodic)
        
        # Calculate hexagonal node positions
        node_positions = {}
        for node_idx in graph.node_indices():
            # For hexagonal lattice, calculate proper hexagonal coordinates
            row = node_idx // cols
            col = node_idx % cols
            
            # Hexagonal lattice positioning
            x = col * 1.5
            y = row * np.sqrt(3) + (col % 2) * np.sqrt(3) / 2
            
            node_positions[node_idx] = (x, y)
        
        info = LatticeInfo(
            lattice_type="hexagonal",
            num_nodes=graph.num_nodes(),
            num_edges=graph.num_edges(),
            shape=(rows, cols),
            node_positions=node_positions,
            parameters={"periodic": periodic}
        )
        
        self.last_generated_info = info
        return graph, info
    
    def generate_triangular_lattice(self, rows: int, cols: int) -> Tuple[rx.PyGraph, LatticeInfo]:
        """
        Generate a triangular lattice by adding diagonal connections to square lattice.
        
        Args:
            rows: Number of rows in the lattice
            cols: Number of columns in the lattice
        
        Returns:
            Tuple of (graph, lattice_info)
        """
        # Start with square lattice
        graph = rx.generators.grid_graph(rows, cols, periodic=False)
        
        # Add diagonal connections to make it triangular
        for i in range(rows - 1):
            for j in range(cols - 1):
                # Add diagonal edges
                node1 = i * cols + j
                node2 = (i + 1) * cols + (j + 1)
                
                if graph.has_node(node1) and graph.has_node(node2):
                    graph.add_edge(node1, node2, None)
        
        # Calculate node positions
        node_positions = {}
        for node_idx in graph.node_indices():
            row = node_idx // cols
            col = node_idx % cols
            node_positions[node_idx] = (float(col), float(row))
        
        info = LatticeInfo(
            lattice_type="triangular",
            num_nodes=graph.num_nodes(),
            num_edges=graph.num_edges(),
            shape=(rows, cols),
            node_positions=node_positions,
            parameters={}
        )
        
        self.last_generated_info = info
        return graph, info
    
    def generate_king_lattice(self, rows: int, cols: int) -> Tuple[rx.PyGraph, LatticeInfo]:
        """
        Generate a king's lattice (square lattice with diagonal connections).
        
        Args:
            rows: Number of rows in the lattice
            cols: Number of columns in the lattice
        
        Returns:
            Tuple of (graph, lattice_info)
        """
        # Start with square lattice
        graph = rx.generators.grid_graph(rows, cols, periodic=False)
        
        # Add all diagonal connections (both directions)
        for i in range(rows):
            for j in range(cols):
                current_node = i * cols + j
                
                # Add diagonal connections if neighbors exist
                directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        neighbor_node = ni * cols + nj
                        if graph.has_node(current_node) and graph.has_node(neighbor_node):
                            if not graph.has_edge(current_node, neighbor_node):
                                graph.add_edge(current_node, neighbor_node, None)
        
        # Calculate node positions
        node_positions = {}
        for node_idx in graph.node_indices():
            row = node_idx // cols
            col = node_idx % cols
            node_positions[node_idx] = (float(col), float(row))
        
        info = LatticeInfo(
            lattice_type="king",
            num_nodes=graph.num_nodes(),
            num_edges=graph.num_edges(),
            shape=(rows, cols),
            node_positions=node_positions,
            parameters={}
        )
        
        self.last_generated_info = info
        return graph, info
    
    def get_node_connectivity(self, graph: rx.PyGraph, node_idx: int) -> List[int]:
        """Get the list of nodes connected to a given node."""
        return [edge[1] for edge in graph.out_edges(node_idx)]
    

    def get_lattice_info(self) -> Optional[LatticeInfo]:
        """Get information about the last generated lattice."""
        return self.last_generated_info
    
    def export_positions_for_visualization(self, graph: rx.PyGraph, info: LatticeInfo) -> Dict[str, Union[List, Dict]]:
        """
        Export lattice data in a format suitable for visualization.
        
        Returns:
            Dictionary with node positions and edge connections for plotting
        """
        edges = [(edge[0], edge[1]) for edge in graph.edge_list()]
        
        return {
            "node_positions": info.node_positions,
            "edges": edges,
            "lattice_type": info.lattice_type,
            "shape": info.shape,
            "num_nodes": info.num_nodes,
            "num_edges": info.num_edges
        }


def analyze_graph_properties(graph: rx.PyGraph) -> Dict:
    """
    Analyze key properties of the QPU connectivity graph.
    
    Returns:
        dict: Graph properties including degrees, chromatic bounds, etc.
    """
    properties = {}
    
    # Calculate node degrees
    node_degrees = {}
    for i, node_att in enumerate(graph.nodes()):
        # Handle case where nodes might not have 'name' attribute
        node_label = node_att['name'] if isinstance(node_att, dict) and 'name' in node_att else str(i)
        degree = graph.degree(i)
        node_degrees[node_label] = degree
    
    properties['node_degrees'] = node_degrees
    if node_degrees:
        properties['max_degree'] = max(node_degrees.values())
        properties['min_degree'] = min(node_degrees.values())
        
        # Calculate chromatic bounds
        # Lower bound: chromatic number ≥ max_degree (for general graphs)
        # Upper bound: chromatic number ≤ max_degree + 1 (Brook's theorem)
        properties['chromatic_lower_bound'] = properties['max_degree']
        properties['chromatic_upper_bound'] = properties['max_degree'] + 1
    
    return properties


from .coloring import RelaxedColoring

def assign_families_graph_coloring(graph: rx.PyGraph, available_families: Dict, strategy: Optional[object] = None) -> Dict:
    """
    Assign frequency families to qubits using graph coloring principles.
    
    Args:
        graph: RustworkX graph representing QPU connectivity
        available_families: List of available frequency families
        strategy: Optional coloring strategy object (callable)
        
    Returns:
        dict: Mapping from qubit names to family assignments
    """
    family_names = list(available_families.keys())
    color_list = [family['color'] for family in available_families.values()]
    
    # Analyze graph structure
    props = analyze_graph_properties(graph)
    
    print(f"Graph Analysis:")
    print(f"  Nodes: {len(graph.nodes())}")
    print(f"  Edges: {len(graph.edges())}")
    
    if strategy is not None:
        # Use the provided strategy (callable)
        color_map = strategy(graph)
        print(f"  Automated coloring used: {type(strategy).__name__}")
        
        # Map integer colors to families
        # Note: This assumes the number of colors found <= number of families
        coloring = {}
        for node_idx, color_idx in color_map.items():
            node_name = graph.nodes()[node_idx]['name']
            # Wrap around if we have more colors than families (or handle error)
            fam_idx = color_idx % len(color_list)
            coloring[node_name] = color_list[fam_idx]
            
    else:
        # Default to RelaxedColoring if no strategy provided
        print("  Using default RelaxedColoring strategy")
        default_strategy = RelaxedColoring()
        color_map = default_strategy(graph)
        
        coloring = {}
        for node_idx, color_idx in color_map.items():
            node_name = graph.nodes()[node_idx]['name']
            fam_idx = color_idx % len(color_list)
            coloring[node_name] = color_list[fam_idx]

    return coloring


def verify_coloring(graph, coloring):
    """
    Verify that the graph coloring is valid (no adjacent nodes have same color).
    
    Args:
        graph: RustworkX graph
        coloring: Dict mapping node indices to colors
        
    Returns:
        bool: True if coloring is valid
    """
    for edge in graph.edge_list():
        node1, node2 = edge
        if coloring.get(node1) == coloring.get(node2):
            return False
    return True


def assign_frequencies_by_topology(labels, families, base_freqs, graph, step=100e6):
    """
    Assigns qubit frequencies based on topology to avoid collisions.
    Rule: If two qubits of the same family share a neighbor, their frequencies must be different by 'step'.

    Args:
        labels:
        families:
        base_freqs:
        graph:
        step:
    Retruns:
        freq_map
    """
    freq_map = {}
    # Start by assigning base frequencies
    for label in labels:
        freq_map[label] = base_freqs[families[label]]

    new_freq_map = {}
    # Check for same-family neighbors and adjust frequencies
    for center_idx in range(len(graph.nodes())):
        neighbor_indices = graph.neighbors(center_idx)
        neighbor_labels = [graph.nodes()[idx]['name'] for idx in neighbor_indices]
        
        # Group neighbors by family
        neighbors_by_family = {}
        for neighbor in neighbor_labels:
            family = families[neighbor]
            if family not in neighbors_by_family:
                neighbors_by_family[family] = []
            neighbors_by_family[family].append(neighbor)
        
        # For each family group with multiple members, ensure unique frequencies
        for family, group in neighbors_by_family.items():
            if len(group) > 1:
                # Sort group to have consistent assignment
                s_group = sorted(group)
                base_freq = base_freqs[family]

                # Remove already updated frequencies and set their frequency as base
                for i, qubit_label in enumerate(s_group):
                    if qubit_label in new_freq_map:
                        base_freq = max(new_freq_map[qubit_label], base_freq)
                        s_group.remove(qubit_label)

                print(f"Adjusting frequencies for family '{family}' neighbors: {s_group} with base freq {base_freq/1e9} GHz")    
                # Assign frequencies for remaining qubits with step increments
                for i, qubit_label in enumerate(s_group): 
                    if qubit_label not in new_freq_map:
                        new_freq_map[qubit_label] = base_freq + (i) * step

                

    # Update freq_map with new assignments
    freq_map.update(new_freq_map)
                    
    return freq_map


def relaxed_coloring(graph: rx.PyGraph, max_same_color_neighbors: int = 2) -> Dict[int, int]:
    """
    Deprecated: Use qfoundry.coloring.RelaxedColoring instead.
    """
    from .coloring import RelaxedColoring
    strategy = RelaxedColoring(max_same_color_neighbors)
    return strategy(graph)
