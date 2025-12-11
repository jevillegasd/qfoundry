import json
import re
import rustworkx as rx
from dataclasses import fields
from typing import Dict, Any, Optional, Protocol, List, Union
from .schema import (
    DesignSpecification, LayoutGraph, ReadoutConfiguration, ComponentParameters,
    LayoutNode, LayoutEdge, ReadoutLine, ReadoutResonatorAssignment, QubitFamily,
)
from .prompts import SPECIFICATION_PROMPT, LAYOUT_PROMPT, READOUT_PROMPT, COMPONENT_PROMPT

class ModelClient(Protocol):
    def generate_content(self, prompt: str) -> Any:
        ...

class QFoundryAgent:
    """
    AI Agent for QFoundry to suggest quantum processor designs based on constraints.
    Supports multi-step generation: Specification -> Layout -> Readout -> Components.
    """
    
    def __init__(self, model_client: Optional[Any] = None, api_key: Optional[str] = None, model_name: str = 'gemini-2.5-flash'):
        """
        Initialize the agent.
        
        Args:
            model_client: An optional object with a `generate_content(prompt)` method.
                          If None, tries to initialize google.generativeai.
            api_key: API key for the internal model client (if model_client is None).
            model_name: Name of the model to use (default: 'gemini-pro').
        """
        self.history: List[Dict[str, str]] = []
        self.current_spec: Optional[DesignSpecification] = None
        self.current_layout: Optional[LayoutGraph] = None
        self._last_response_: Any = None
        self._last_query_: Any = None
        
        if model_client:
            self.client = model_client
        else:
            try:
                import google.generativeai as genai
                if api_key:
                    genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(model_name)
            except ImportError:
                raise ImportError("google-generativeai is not installed. Please install it or provide a custom model_client.")
            except Exception as e:
                raise ValueError(f"Failed to initialize Gemini client: {e}")

    def history_text(self) -> str:
        """Convert history to a formatted string for prompts."""
        history_text = ""
        for entry in self.history:
            history_text += f"User: {entry['user']}\nAgent: {entry['agent']}\n"
        return history_text


    def _query_model(self, system_prompt: str, user_input: str) -> Dict[str, Any]:
        """Helper to query the model and parse JSON response."""
        # Build prompt with history (optional, maybe only relevant for spec generation)
        # For specific tasks, history might be less relevant or need to be handled differently
        
        full_prompt = f"{system_prompt}\n\nUser Input: {user_input}\n\nJSON Response:"
        
        response = self.client.generate_content(full_prompt)
        
        if hasattr(response, 'text'):
            response_text = response.text
        else:
            response_text = str(response)
        self._last_query_ = full_prompt
        self._last_response_ = response_text
        
        json_match = re.search(r"```json(.*?)```", response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)

        # Basic cleanup
        clean_text = response_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
            
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse model response as JSON: {e}\nResponse: {response_text}")

    def suggest_design(self, user_constraints: str) -> Dict[str, Any]:
        """
        Legacy method for backward compatibility. Generates specification only.
        """
        return self.generate_specification(user_constraints)

    def generate_specification(self, user_constraints: str) -> Dict[str, Any]:
        """
        Generate a high-level design specification.
        """
        # Include history for conversational refinement
        prompt = SPECIFICATION_PROMPT.format() + "\n\n" + self.history_text()
        
        result = self._query_model(prompt, user_constraints)
        
        # Update history
        self.history.append({
            "user": user_constraints,
            "agent": json.dumps(result)
        })
        
        # Parse nested objects for DesignSpecification
        spec_data = result.copy()
        try:
            families = [QubitFamily(**fam) for fam in result.get('qubit_families', [])]
            spec_data['qubit_families'] = families
        except Exception as e:
            print(f"\033[93mWarning: Failed to parse QubitFamily object: {e}\033[0m")
            spec_data = result.copy()
            # Fallback: Initializze thge Specifiction and accept   (qubit families will be dicts)
        
        try:
            self.current_spec = DesignSpecification(**spec_data)
        except Exception as e:
            print(f"\033[93mWarning: Failed to parse DesignSpecification object: {e}\033[0m")
            self.current_spec = spec_data # Fallback: keep as dict, this may be problematic elsewhere
        
        return result

    def generate_layout(self, specification: Dict[str, Any] = None) -> Optional[LayoutGraph]:
        """
        Generate the physical layout graph (nodes, edges, frequencies).
        """
        if specification is None:
            if self.current_spec is None:
                raise ValueError("No specification provided and no current specification available.")
            specification = self.current_spec.serialize()
        
        else:
            if type(specification) is DesignSpecification:
                # When a new specification is provided as a DesignSpecification object, this is prioritized
                specification = specification.serialize()
            else:
                # Otherwiuse is appended to the current specification
                specification = {
                'current_specs' : self.current_spec.serialize() ,
                'additional_specs' : specification
                }

        spec_str = json.dumps(specification, indent=2)
        result = self._query_model(LAYOUT_PROMPT.format(specification=spec_str), "Generate layout based on this specification.")

        self.current_layout = self.parse_layout_response(result)
        return self.current_layout

    def parse_layout_response(self, result: Dict[str, Any]) -> Optional[LayoutGraph]:
        """
        Parse nested objects for LayoutGraph, filtering unknown fields.
        """
        try:
            # Helper to filter dict keys based on dataclass fields
            def filter_kwargs(cls, data):
                valid_keys = {f.name for f in fields(cls)}
                return {k: v for k, v in data.items() if k in valid_keys}

            nodes = [LayoutNode(**filter_kwargs(LayoutNode, node)) for node in result.get('nodes', [])]
            edges = [LayoutEdge(**filter_kwargs(LayoutEdge, edge)) for edge in result.get('edges', [])]
            
            return LayoutGraph(nodes=nodes, edges=edges)
        except Exception as e:
            print(f"Warning: Failed to parse LayoutGraph object: {e}")
            return None
    

    def get_rustworkx_graph(self) -> Optional[rx.PyGraph]:
        """
        Convert the current LayoutGraph to a rustworkx PyGraph.
        
        Returns:
            rx.PyGraph: A graph where node payloads are LayoutNode objects (or dicts)
                        and edge payloads are LayoutEdge objects (or dicts).
        """
        if not self.current_layout:
            return None
            
        graph = rx.PyGraph()
        
        # Map node IDs to graph indices
        id_to_idx = {}
        
        for node in self.current_layout.nodes:
            # Add node to graph, payload is the node object itself
            idx = graph.add_node(node)
            id_to_idx[node.id] = idx
            
        for edge in self.current_layout.edges:
            if edge.source in id_to_idx and edge.target in id_to_idx:
                source_idx = id_to_idx[edge.source]
                target_idx = id_to_idx[edge.target]
                graph.add_edge(source_idx, target_idx, edge)
                
        return graph

    def update_layout_from_graph(self, graph: rx.PyGraph):
        """
        Update the agent's current layout from a rustworkx graph.
        
        Args:
            graph: A rustworkx PyGraph. Nodes should ideally be LayoutNode objects or dicts.
                   Edges should be LayoutEdge objects or dicts.
        """
        nodes = []
        edges = []
        
        # Extract nodes
        for node_payload in graph.nodes():
            if isinstance(node_payload, LayoutNode):
                nodes.append(node_payload)
            elif isinstance(node_payload, dict):
                nodes.append(LayoutNode(**node_payload))
            else:
                # Handle unknown payload type, maybe create a generic node
                # Assuming payload might be just a name or ID string
                nodes.append(LayoutNode(id=str(node_payload), type='qubit'))

        # Extract edges
        for source_idx, target_idx, edge_payload in graph.edge_index_map().values():
            source_node = graph.get_node_data(source_idx)
            target_node = graph.get_node_data(target_idx)
            
            # Get IDs safely
            source_id = source_node.id if isinstance(source_node, LayoutNode) else (source_node.get('id') if isinstance(source_node, dict) else str(source_node))
            target_id = target_node.id if isinstance(target_node, LayoutNode) else (target_node.get('id') if isinstance(target_node, dict) else str(target_node))
            
            if isinstance(edge_payload, LayoutEdge):
                edges.append(edge_payload)
            elif isinstance(edge_payload, dict):
                edges.append(LayoutEdge(**edge_payload))
            else:
                # Default edge
                edges.append(LayoutEdge(source=source_id, target=target_id, type='capacitive'))
                
        self.current_layout = LayoutGraph(nodes=nodes, edges=edges)

    def generate_readout_scheme(self, layout: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate readout multiplexing scheme.
        """
        if layout is None:
            layout = self.current_layout.serialize()
        # Summarize layout for prompt to save tokens
        nodes_summary = [{"id": n["id"], "frequency": n.get("frequency")} for n in layout.get("nodes", [])]
        edges_summary = [{"source": e["source"], "target": e["target"]} for e in layout.get("edges", [])]
        layout_summary = json.dumps({"nodes": nodes_summary, "edges": edges_summary}, indent=2)
        
        query = READOUT_PROMPT.format(layout_summary=layout_summary)
        result = self._query_model(query, "Generate readout scheme.")

        return result

    def generate_component(self, specification: Dict[str, Any], node_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate detailed parameters for a single component.
        """
        spec_str = json.dumps(specification, indent=2)
        node_str = json.dumps(node_info, indent=2)
        
        result = self._query_model(COMPONENT_PROMPT.format(specification=spec_str, node_info=node_str), "Calculate component parameters.")

        return result

