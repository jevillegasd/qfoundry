import rustworkx as rx
from typing import Dict, List, Optional

class RelaxedColoring:
    """
    A custom coloring strategy for RustworkX that implements relaxed constraints.
    
    This strategy enforces two rules:
    1. Standard Coloring: A node cannot have the same color as its neighbors.
    2. Neighborhood Constraint: A node cannot have more than N neighbors of the same color.
       This is useful for frequency allocation where we want to limit cross-talk.
    """
    
    def __init__(self, max_same_color_neighbors: int = 2):
        self.max_same_color_neighbors = max_same_color_neighbors

    def __call__(self, graph: rx.PyGraph) -> Dict[int, int]:
        nodes = sorted(graph.node_indices(), key=lambda n: graph.degree(n), reverse=True)
        colors = {}
        
        for node in nodes:
            color = 0
            while True:
                is_valid = True
                
                neighbors = graph.neighbors(node)
                
                for neighbor in neighbors:
                    if neighbor in colors and colors[neighbor] == color:
                        is_valid = False
                        break
                
                if is_valid:
                    for neighbor in neighbors:
                        neighbor_neighbors = graph.neighbors(neighbor)
                        neighbor_same_color_count = 0
                        for nn in neighbor_neighbors:
                            if nn in colors and colors[nn] == color:
                                neighbor_same_color_count += 1
                        
                        if neighbor_same_color_count >= self.max_same_color_neighbors:
                            is_valid = False
                            break
                
                if is_valid:
                    colors[node] = color
                    break
                
                color += 1
                
        return colors
