import json
import re
import os
import rustworkx as rx
from dataclasses import fields
from typing import Dict, Any, Optional, Protocol, List, Union, Tuple
from .schema import (
    DesignSpecification, LayoutGraph, ReadoutConfiguration, ComponentParameters,
    LayoutNode, LayoutEdge, ReadoutLine, ReadoutResonatorAssignment, QubitFamily,
)
from .prompts import SPECIFICATION_PROMPT, LAYOUT_PROMPT, READOUT_PROMPT, COMPONENT_PROMPT
from rustworkx.visualization import mpl_draw

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
        self.system_prompt_file = "agent_system_prompt.txt"
        
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

    def reset(self):
        """Reset the agent's state (history, spec, layout)."""
        self.history = []
        self.current_spec = None
        self.current_layout = None
        self._last_response_ = None

    def chat(self, user_input: str, stream: bool = False) -> Union[str, Any]:
        """
        Chat with the agent about the current design.
        Updates history but does not modify spec/layout directly.
        
        Args:
            user_input: The user's message.
            stream: If True, returns a generator yielding response chunks.
        """
        system_prompt = self._load_system_prompt()
        context = self._get_context_text()
        history = self.history_text()
        
        prompt = f"{system_prompt}\n\nContext:\n{context}\n\nHistory:\n{history}\n\nUser Input: {user_input}"
        
        if stream:
            response = self.client.generate_content(prompt, stream=True)
            
            def stream_wrapper():
                full_text = ""
                for chunk in response:
                    text = chunk.text if hasattr(chunk, 'text') else str(chunk)
                    full_text += text
                    yield text
                
                self._last_response_ = full_text
                self.history.append({
                    "user": user_input,
                    "agent": full_text
                })
            
            return stream_wrapper()
        else:
            response = self.client.generate_content(prompt)
            
            if hasattr(response, 'text'):
                response_text = response.text
            else:
                response_text = str(response)
                
            self._last_response_ = response_text
            
            self.history.append({
                "user": user_input,
                "agent": response_text
            })
            
            return response_text

    def _load_system_prompt(self) -> str:
        """Load the system prompt from the local file."""
        if os.path.exists(self.system_prompt_file):
            try:
                with open(self.system_prompt_file, "r") as f:
                    return f.read().strip()
            except Exception as e:
                print(f"Warning: Failed to read system prompt file: {e}")
        return "You are an expert superconducting quantum processor designer."

    def _get_context_text(self) -> str:
        """Get the current design context (spec, layout) as a string."""
        context = ""
        if self.current_spec:
            context += f"\nCurrent Specification:\n{json.dumps(self.current_spec.serialize(), indent=2)}\n"
        if self.current_layout:
            # Serialize layout, potentially summarizing if too large in future
            context += f"\nCurrent Layout:\n{json.dumps(self.current_layout.serialize(), indent=2)}\n"
        return context

    def history_text(self) -> str:
        """Convert history to a formatted string for prompts."""
        history_text = ""
        for entry in self.history:
            history_text += f"User: {entry['user']}\nAgent: {entry['agent']}\n"
        return history_text


    def _query_model(self, system_prompt: str, user_input: str) -> Tuple[Dict[str, Any], str]:
        """Helper to query the model and parse JSON response. Returns (json_data, explanation)."""
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
        
        json_data = {}
        explanation = ""
        
        # Extract JSON block
        json_match = re.search(r"```json(.*?)```", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            explanation = response_text.replace(json_match.group(0), "").strip()
        else:
            # Attempt to find the first '{' and last '}'
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx+1]
                explanation = (response_text[:start_idx] + response_text[end_idx+1:]).strip()
            else:
                # Fallback: try to parse the whole text as JSON
                json_str = response_text
                explanation = ""

        # Clean up json_str
        clean_json = json_str.strip()
        if clean_json.startswith("```json"):
            clean_json = clean_json[7:]
        if clean_json.startswith("```"):
            clean_json = clean_json[3:]
        if clean_json.endswith("```"):
            clean_json = clean_json[:-3]
            
        try:
            json_data = json.loads(clean_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse model response as JSON: {e}\nResponse: {response_text}")
            
        return json_data, explanation

    def suggest_design(self, user_constraints: str) -> Dict[str, Any]:
        """
        Legacy method for backward compatibility. Generates specification only.
        """
        return self.generate_specification(user_constraints)

    def generate_specification(self, user_constraints: str) -> Tuple[Dict[str, Any], str]:
        """
        Generate a high-level design specification.
        """
        # Include history for conversational refinement
        system_prompt = self._load_system_prompt()
        context = self._get_context_text()
        history = self.history_text()
        
        prompt = f"{system_prompt}\n\n{SPECIFICATION_PROMPT}\n\nContext:\n{context}\n\nHistory:\n{history}"
        
        result, explanation = self._query_model(prompt, user_constraints)
        
        # Update history
        self.history.append({
            "user": user_constraints,
            "agent": self._last_response_
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
        
        return result, explanation

    def generate_layout(self, specification: Dict[str, Any] = None) -> Tuple[Optional[LayoutGraph], str]:
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
        
        system_prompt = self._load_system_prompt()
        context = self._get_context_text()
        history = self.history_text()
        
        prompt = f"{system_prompt}\n\n{LAYOUT_PROMPT.format(specification=spec_str)}\n\nContext:\n{context}\n\nHistory:\n{history}"
        
        result, explanation = self._query_model(prompt, "Generate layout based on this specification.")

        self.current_layout = self.parse_layout_response(result)
        return self.current_layout, explanation

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
        else :
            return self.current_layout.toRx()
                
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

    def generate_readout_scheme(self, layout: Dict[str, Any] = None) -> Tuple[Dict[str, Any], str]:
        """
        Generate readout multiplexing scheme.
        """
        if layout is None:
            layout = self.current_layout.serialize()
        # Summarize layout for prompt to save tokens
        nodes_summary = [{"id": n["id"], "frequency": n.get("frequency")} for n in layout.get("nodes", [])]
        edges_summary = [{"source": e["source"], "target": e["target"]} for e in layout.get("edges", [])]
        layout_summary = json.dumps({"nodes": nodes_summary, "edges": edges_summary}, indent=2)
        
        system_prompt = self._load_system_prompt()
        context = self._get_context_text()
        history = self.history_text()
        
        query = f"{system_prompt}\n\n{READOUT_PROMPT.format(layout_summary=layout_summary)}\n\nContext:\n{context}\n\nHistory:\n{history}"
        
        result, explanation = self._query_model(query, "Generate readout scheme.")
        return result, explanation

    def generate_component(self, specification: Dict[str, Any], node_info: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """
        Generate detailed parameters for a single component.
        """
        spec_str = json.dumps(specification, indent=2)
        node_str = json.dumps(node_info, indent=2)
        
        system_prompt = self._load_system_prompt()
        context = self._get_context_text()
        history = self.history_text()
        
        prompt = f"{system_prompt}\n\n{COMPONENT_PROMPT.format(specification=spec_str, node_info=node_str)}\n\nContext:\n{context}\n\nHistory:\n{history}"
        
        result, explanation = self._query_model(prompt, "Calculate component parameters.")
        return result, explanation

    def draw_layout(self, seed: Optional[int] = 8):
        """ Helper fucntion to draw the circuit layout using rustowrks"""
        layout = self.current_layout
        if layout is None:
            raise ValueError("No current layout to draw.")
        
        graph = layout.toRx()
        pos = rx.spring_layout(graph, k=0.3, seed = seed)
        colors = [node.color for node in layout.nodes]
        def labels_func(node):
            return f"{node.type[0].upper()}{node.id}\nFreq:{node.frequency/1e9:.2f}GHz"

        fig = mpl_draw(graph, pos= pos, with_labels=True, labels=labels_func, node_size=500, font_size=8, node_color= colors)
        return fig