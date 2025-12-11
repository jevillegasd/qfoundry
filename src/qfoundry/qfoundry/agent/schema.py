from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Literal
from dataclasses import asdict

@dataclass
class SerializableMixin:
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the dataclass to a nested dictionary.
        """
        return asdict(self)

@dataclass
class QubitFamily(SerializableMixin):
    family_id: str
    qubit_type: str
    frequency_mean: float
    anharmonicity: float
    asymmetry_max: float
    charging_energy: float
    drive_line: bool = False
    drive_coupling: Optional[float] = None
    readout_resonator: bool = False
    resonator_coupling: Optional[float] = None
    color: Optional[int] = None  # for frequency grouping and visualization

@dataclass
class DesignSpecification(SerializableMixin):
    design_name: str
    description: str
    qubit_count: int
    topology_type: str
    grid_dimensions: Optional[List[int]]
    qubit_families: List[QubitFamily]
    coupling_type: str
    coupling_strength_mean: float
    coloring_strategy: str
    design_guidelines: List[str]

@dataclass
class LayoutNode(SerializableMixin):
    id: str
    type: Literal['qubit', 'coupler', 'resonator']
    family_id: Optional[str] = None
    position: Optional[List[float]] = None
    frequency: Optional[float] = None
    color: Optional[int] = None

@dataclass
class LayoutEdge(SerializableMixin):
    source: str
    target: str
    type: str

@dataclass
class LayoutGraph(SerializableMixin):
    nodes: List[LayoutNode]
    edges: List[LayoutEdge]

@dataclass
class ReadoutResonatorAssignment(SerializableMixin):
    qubit_id: str
    resonator_frequency: float
    coupling_strength: float

@dataclass
class ReadoutLine(SerializableMixin):
    line_id: str
    assignments: List[ReadoutResonatorAssignment]

@dataclass
class ReadoutConfiguration(SerializableMixin):
    lines: List[ReadoutLine]
    multiplexing_guidelines: List[str]

@dataclass
class ComponentParameters(SerializableMixin):
    component_id: str
    component_type: str
    parameters: Dict[str, Any]
    metrics: Optional[Dict[str, Any]] = None

@dataclass
class FullDesign(SerializableMixin):
    specification: DesignSpecification
    layout: LayoutGraph
    readout: ReadoutConfiguration
    components: List[ComponentParameters]
